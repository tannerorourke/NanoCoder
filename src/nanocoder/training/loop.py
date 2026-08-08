""" Pretraining loop + loss estimation, live plotting """

import os
import time

import torch

from nanocoder.data.corpus import get_batch
from nanocoder.training.checkpoint import rotate_checkpoints, save_checkpoint

CKPT_PREFIX = "nanocoder"


def _fmt_eta(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def update_plots(c_it, train_log, val_log, lr_log, gn_log, plot_handle, max_iters, base_lr):
    # bail before importing matplotlib it when there is no handle to draw into.
    if plot_handle is None:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plots = axes.flatten()
    fig.suptitle(f'Training progress | iter {c_it}/{max_iters}', fontsize=11)

    ax = plots[0]
    if train_log:
        tl_it, tl_val = zip(*train_log)
        ax.plot(tl_it, tl_val, marker='o', label='train', markersize=3)
    if val_log:
        vl_it, vl_val = zip(*val_log)
        ax.plot(vl_it, vl_val, marker='o', label='val', linewidth=1.2, markersize=5)
    ax.set_ylabel('Cross-Entropy loss', fontsize=9)
    ax.set_title('Loss')
    ax.set_ylim(bottom=0)
    ax.legend()

    ax = plots[1]
    if lr_log:
        lr_it, lr_val = zip(*lr_log)
        ax.plot(lr_it, lr_val, marker='o', color='C2', linewidth=1.2)
    ax.set_ylabel('rate', fontsize=9)
    ax.set_title('LR schedule')
    ax.set_ylim(bottom=0, top=base_lr * 1.1)

    ax = plots[2]
    if gn_log:
        gn_it, gn_val = zip(*gn_log)
        ax.plot(gn_it, gn_val, color='C3', linewidth=0.8)
    ax.set_ylabel(r'$\|$Grad$\|_2$', fontsize=9)
    ax.set_title('Gradient norm')
    ax.set_ylim(bottom=0)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0, right=max_iters)
    plt.tight_layout()
    if plot_handle is not None:
        plot_handle.update(fig)
    plt.close(fig)


def train_loop(
    model, optimizer, scheduler,
    data_tr, data_val,
    block_size: int, batch_size: int,
    tcfg,
    device: str,
    autocast_ctx,
    scaler=None,
    plot_handle=None,
    start_iter: int = 0,
    checkpoint_dir: str = "./checkpoints",
):
    """
    Run to tcfg.max_iters, resuming from start_iter when given.

    get_batch draws random offsets, so there is no sampler position to restore; a resume is
    fully described by the model, optimizer and scheduler state.
    """
    use_scaler = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ('train', 'val'):
            losses = torch.zeros(tcfg.eval_iters)
            for k in range(tcfg.eval_iters):
                X, Y = get_batch(data_tr if split == 'train' else data_val,
                                 block_size, batch_size, device)
                with autocast_ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    forward = torch.compile(model)
    model.train()

    train_log, val_log, lr_log, gn_log = [], [], [], []

    X, Y = get_batch(data_tr, block_size, batch_size, device)  # prefetch first batch
    ema_dt = None
    t_iter = time.time()
    for it in range(start_iter + 1, tcfg.max_iters + 1):
        # -- accumulated on-device. Calling .item() per micro-step would sync the GPU
        #    grad_accum_steps times an iteration purely to build a log line.
        total_loss = torch.zeros((), device=device)

        for micro_step in range(tcfg.grad_accum_steps):
            with autocast_ctx:
                _, loss = forward(X, Y)
                loss = loss / tcfg.grad_accum_steps
            total_loss += loss.detach()

            X, Y = get_batch(data_tr, block_size, batch_size, device)  # async prefetch
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        gn_val = gn.item()
        gn_log.append((it, gn_val))
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # -- the iteration after a start or resume pays for torch.compile, so it stays out
        #    of the average the ETA reads from
        dt = time.time() - t_iter
        t_iter = time.time()
        if it > start_iter + 1:
            ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt

        tl = total_loss.item()          # -- one sync per iteration
        lr = scheduler.last_lr
        is_log_it = it % tcfg.log_interval == 0
        is_val_it = it % tcfg.eval_interval == 0
        is_ckpt_it = it % tcfg.checkpoint_interval == 0
        is_last = it == tcfg.max_iters

        summ = f"\riter {it:<5}/{tcfg.max_iters} | Tl: {tl:.4f}"
        if is_log_it or is_last:
            train_log.append((it, tl))
            lr_log.append((it, lr))
            update_plots(it, train_log, val_log, lr_log, gn_log, plot_handle,
                         tcfg.max_iters, tcfg.base_lr)

        if is_val_it or is_last:
            losses = estimate_loss()
            val_log.append((it, losses['val']))
            summ += f" | Vl: {losses['val']:.4f}"

        summ += f" | lr:{lr:.2e} | gn:{gn_val:.3f}"
        if ema_dt:
            summ += f" | {ema_dt:.2f}s/it | ETA {_fmt_eta((tcfg.max_iters - it) * ema_dt)}"
        # -- sole writer of this line. A second print would overwrite it within the same
        #    iteration, which is what kept the old rate counter off screen.
        print(summ, end='\n' if (is_log_it or is_val_it or is_ckpt_it or is_last) else '',
              flush=True)

        if is_ckpt_it or is_last:
            path = save_checkpoint(os.path.join(checkpoint_dir, f"{CKPT_PREFIX}_{it}.pt"),
                                   model, optimizer, step=it, scaler=scaler)
            rotate_checkpoints(checkpoint_dir, CKPT_PREFIX, tcfg.keep_checkpoints)
            print(f"Saved {path}", flush=True)

    model.eval()
    return model, train_log, val_log, lr_log, gn_log
