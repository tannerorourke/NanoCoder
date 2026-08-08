"""
Direct Preference Optimization

Instead of a scalar reward model, recognises the optimal policy under a KL-constrained 
reward objective has a closed form, inverts that to express the reward *as* a log-ratio 
between policy and reference, and substitute straight into Bradley-Terry preference likelihood
where each term is a summed completion log-prob:

    loss = -log sigmoid(beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

- bracketed quantity: The margin, how much more the policy prefers the chosen completion than 
the reference does, minus the same for the rejected one. The reference never updates, so it acts 
purely as an anchor - 
- beta: how far the policy may drift from it.

Two things easy to get (quietly) wrongg:

- **Log-probs are summed over completion tokens only.** The prompt is identical on both
  sides, so including it adds the same constant to chosen and rejected and cancels from the
  margin - but only if the masking is exact. The mask comes from data/sft.py unchanged.
- **Loss alone is a poor diagnostic.** A falling loss is compatible with the policy driving
  *both* log-probs down and merely separating them, which is degenerate --> log both of the 
  implicit rewards, their margin, and chosen/rejected accuracy are all logged.
"""
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from nanocoder.constants import SEED
from nanocoder.data.preference import pair_batches
from nanocoder.data.sft import IGNORE
from nanocoder.training.checkpoint import rotate_checkpoints, save_checkpoint
from nanocoder.training.sft_loop import constant_with_warmup


def sequence_logprobs(model, x, y, autocast_ctx=None):
    """
    Summed log-probability of each row's completion tokens.
    - y carries -1 on prompt and padding positions, so the mask is already computed
    """
    with (autocast_ctx if autocast_ctx is not None else nullcontext()):
        logits, _ = model(x, y)             # the supervised path already returns all logits
    logits = logits.float()

    mask = y != IGNORE
    safe_y = y.masked_fill(~mask, 0)        # gather needs a valid index everywhere
    token_lp = torch.log_softmax(logits, dim=-1).gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(-1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float):
    """ The Bradley-Terry objective over implicit rewards, plus its diagnostics """
    r_c = beta * (pi_c - ref_c)
    r_r = beta * (pi_r - ref_r)
    margin = r_c - r_r
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "reward_chosen": r_c.mean().item(),
        "reward_rejected": r_r.mean().item(),
        "margin": margin.mean().item(),
        "accuracy": (margin > 0).float().mean().item(),
    }


def freeze_reference(model):
    """ Detached, eval-mode copy that never updates """
    import copy
    ref = copy.deepcopy(model)
    ref.eval()
    ref.requires_grad_(False)
    return ref


@torch.no_grad()
def evaluate_pairs(policy, ref, pairs, cfg, eos_id, device, autocast_ctx, max_batches):
    """Loss and diagnostics over held-out pairs."""
    policy.eval()
    acc, agg = 0, {"loss": 0.0, "margin": 0.0, "accuracy": 0.0}
    autocast_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    for i, (x, y, n) in enumerate(pair_batches(pairs, cfg.block_size, cfg.batch_size,
                                               eos_id, device, shuffle=False)):
        if i >= max_batches:
            break
        pi = sequence_logprobs(policy, x, y, autocast_ctx)
        rf = sequence_logprobs(ref, x, y, autocast_ctx)
        loss, m = dpo_loss(pi[:n], pi[n:], rf[:n], rf[n:], cfg.beta)
        agg["loss"] += loss.item()
        agg["margin"] += m["margin"]
        agg["accuracy"] += m["accuracy"]
        acc += 1
    policy.train()
    return {k: v / max(1, acc) for k, v in agg.items()}


def dpo_loop(policy, ref, optimizer, train_pairs, val_pairs, cfg, eos_id,
             device, autocast_ctx, scaler=None, checkpoint_dir=None,
             start_step: int = 0, ckpt_prefix: str = "dpo"):
    """
    One pass of preference tuning.
    - Margin rising while accuracy stays flat is the signature of the policy exploiting a 
      surface correlate - length or style - rather than learning the preference. 
    - Both rewards falling steeply means the policy is running away from the reference and 
      beta is too low or the LR too high.
    """
    autocast_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    use_scaler = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
    lr_apply = constant_with_warmup(optimizer, cfg.base_lr, cfg.warmup_steps)
    policy.train()

    steps_per_epoch = max(1, (len(train_pairs) // cfg.batch_size) // cfg.grad_accum_steps)
    start_epoch = start_step // steps_per_epoch
    log, step, t0 = [], start_step, time.time()
    lr_apply(step)      # -- construction applied the warmup rate; a resume is past it

    for epoch in range(start_epoch, cfg.epochs):
        # -- seeded per epoch so a resume rebuilds this order and skips into it
        batches = list(pair_batches(train_pairs, cfg.block_size, cfg.batch_size,
                                    eos_id, device, shuffle=True,
                                    rng=random.Random(SEED + epoch)))
        n_steps = len(batches) // cfg.grad_accum_steps
        first = step - epoch * steps_per_epoch
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}: {len(batches)} batches -> {n_steps} steps"
              + (f", resuming at step {first}" if first else ""))

        for s in range(first, n_steps):
            totals = {"loss": 0.0, "margin": 0.0, "accuracy": 0.0,
                      "reward_chosen": 0.0, "reward_rejected": 0.0}
            for micro in range(cfg.grad_accum_steps):
                x, y, n = batches[s * cfg.grad_accum_steps + micro]
                pi = sequence_logprobs(policy, x, y, autocast_ctx)
                with torch.no_grad():
                    rf = sequence_logprobs(ref, x, y, autocast_ctx)
                loss, m = dpo_loss(pi[:n], pi[n:], rf[:n], rf[n:], cfg.beta)
                loss = loss / cfg.grad_accum_steps

                totals["loss"] += loss.item()
                for k in ("margin", "accuracy", "reward_chosen", "reward_rejected"):
                    totals[k] += m[k] / cfg.grad_accum_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if use_scaler:
                scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            lr = lr_apply(step)
            if step % cfg.log_interval == 0:
                log.append((step, dict(totals)))
                print(f"\rstep {step:<4} | loss {totals['loss']:.4f} | "
                      f"margin {totals['margin']:+.3f} | acc {totals['accuracy']:.3f} | "
                      f"r_c {totals['reward_chosen']:+.3f} r_r {totals['reward_rejected']:+.3f} | "
                      f"lr {lr:.1e} | gn {float(gn):.3f}", flush=True)

            if val_pairs and step % cfg.eval_interval == 0:
                v = evaluate_pairs(policy, ref, val_pairs, cfg, eos_id, device,
                                   autocast_ctx, cfg.eval_batches)
                print(f"\rstep {step:<4} | VAL loss {v['loss']:.4f} | "
                      f"margin {v['margin']:+.3f} | acc {v['accuracy']:.3f}", flush=True)

            # -- only the policy is saved; the reference is rebuilt from the base checkpoint
            if checkpoint_dir and step % cfg.checkpoint_interval == 0:
                path = save_checkpoint(
                    os.path.join(checkpoint_dir, f"{ckpt_prefix}_{step}.pt"),
                    policy, optimizer, step=step, scaler=scaler, extra={"epoch": epoch + 1})
                rotate_checkpoints(checkpoint_dir, ckpt_prefix, cfg.keep_checkpoints)
                print(f"\rSaved {path}", flush=True)

    # -- the interval save already covers a final step that lands on it
    if checkpoint_dir and step % cfg.checkpoint_interval != 0:
        path = save_checkpoint(os.path.join(checkpoint_dir, f"{ckpt_prefix}_{step}.pt"),
                               policy, optimizer, step=step, scaler=scaler,
                               extra={"epoch": cfg.epochs})
        rotate_checkpoints(checkpoint_dir, ckpt_prefix, cfg.keep_checkpoints)
        print(f"Saved {path}")

    if val_pairs:
        v = evaluate_pairs(policy, ref, val_pairs, cfg, eos_id, device,
                           autocast_ctx, cfg.eval_batches)
        print(f"\nFinal validation | loss {v['loss']:.4f} | margin {v['margin']:+.3f} | "
              f"accuracy {v['accuracy']:.3f}")
    print(f"Elapsed {time.time() - t0:.0f}s over {step} steps")

    policy.eval()
    return policy, log
