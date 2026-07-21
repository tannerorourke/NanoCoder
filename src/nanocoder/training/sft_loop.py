"""
Epoch-based fine-tuning loop. Used by both SFT and RFT

Unlike train_loop (pretraining, random offsets from infinite token 
stream), walk a finite list of examples fixed num of times and
"""
import os
import time
from contextlib import nullcontext

import torch

from nanocoder.data.sft import sft_batches
from nanocoder.training.schedule import constant_with_warmup


@torch.no_grad()
def estimate_loss(model, examples, cfg, eos_id, device, autocast_ctx, max_batches: int):
    """Mean completion-masked loss over a fixed slice of the validation split."""
    autocast_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    model.eval()
    losses = []
    for i, (x, y) in enumerate(sft_batches(examples, cfg.block_size, cfg.batch_size,
                                           eos_id, device, shuffle=False)):
        if i >= max_batches:
            break
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def sft_loop(
    model, optimizer,
    train_examples, val_examples,
    cfg, eos_id: int,
    device: str,
    autocast_ctx,
    scaler=None,
    checkpoint_dir: str | None = None,
):
    """
    Run cfg.epochs passes over train_examples.

    The loss is completion-masked upstream: sft_batches emits labels carrying -1 on prompt
    and padding positions, and GPT.forward's cross_entropy already ignores those. So the
    reported loss is over solution tokens only, and is directly comparable across stages -
    unlike a pretrain loss, which averages over prompts the model is never asked to write.
    """
    autocast_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    use_scaler = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
    lr_apply = constant_with_warmup(optimizer, cfg.base_lr, cfg.warmup_steps)
    model.train()

    train_log, val_log, lr_log, gn_log = [], [], [], []
    step = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        batches = list(sft_batches(train_examples, cfg.block_size, cfg.batch_size,
                                   eos_id, device, shuffle=True))
        n_steps = len(batches) // cfg.grad_accum_steps
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}: {len(batches)} batches -> {n_steps} steps")

        for s in range(n_steps):
            total = 0.0
            for micro in range(cfg.grad_accum_steps):
                x, y = batches[s * cfg.grad_accum_steps + micro]
                with autocast_ctx:
                    _, loss = model(x, y)
                    loss = loss / cfg.grad_accum_steps
                total += loss.item()
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if use_scaler:
                scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            lr = lr_apply(step)
            gn_log.append((step, float(gn)))

            if step % cfg.log_interval == 0:
                train_log.append((step, total))
                lr_log.append((step, lr))
                el = time.time() - t0
                print(f"\rstep {step:<5} | Tl: {total:.4f} | lr: {lr:.2e} | "
                      f"gn: {float(gn):.3f} | {el / step:.2f}s/step", flush=True)

            if val_examples and step % cfg.eval_interval == 0:
                vl = estimate_loss(model, val_examples, cfg, eos_id, device,
                                   autocast_ctx, cfg.eval_batches)
                val_log.append((step, vl))
                print(f"\rstep {step:<5} | Vl: {vl:.4f}", flush=True)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"sft_epoch{epoch + 1}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, path)
            print(f"Saved {path}")

    if val_examples:
        vl = estimate_loss(model, val_examples, cfg, eos_id, device,
                           autocast_ctx, cfg.eval_batches)
        val_log.append((step, vl))
        print(f"\nFinal validation loss: {vl:.4f}")

    model.eval()
    return model, train_log, val_log, lr_log, gn_log
