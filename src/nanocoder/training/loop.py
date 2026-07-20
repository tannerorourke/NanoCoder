"""
Training loop, loss estimation, and live plotting.
"""
import os
import time

import torch

from nanocoder.data.corpus import get_batch


def update_plots(c_it, train_log, val_log, lr_log, gn_log, plot_handle, max_iters, base_lr):
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
):
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
    t_iter = time.time()
    for it in range(1, tcfg.max_iters + 1):
        total_loss = 0.0

        # --- accum grads
        for micro_step in range(tcfg.grad_accum_steps):
            with autocast_ctx:
                _, loss = forward(X, Y)
                loss = loss / tcfg.grad_accum_steps

            X, Y = get_batch(data_tr, block_size, batch_size, device)  # async prefetch
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        gn_log.append((it, gn.item()))
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # --- logging
        dt = time.time() - t_iter
        it_per_sec = 1 / dt if dt > 0 else 0
        print(f"\riter {it:<5}/{tcfg.max_iters} -- "
              f"{100 * it / tcfg.max_iters:.2f}% ({it_per_sec:5.2f}it/s)",
              end='', flush=True)
        t_iter = time.time()

        lr = scheduler.last_lr
        summ = f"\riter {it:<5} | Tl: {loss.item() * tcfg.grad_accum_steps:.4f}"
        is_log_it = it % tcfg.log_interval == 0
        is_val_it = it % tcfg.eval_interval == 0
        if is_log_it or it == tcfg.max_iters:
            train_log.append((it, loss.item() * tcfg.grad_accum_steps))
            lr_log.append((it, lr))
            update_plots(it, train_log, val_log, lr_log, gn_log, plot_handle,
                         tcfg.max_iters, tcfg.base_lr)

        if is_val_it or it == tcfg.max_iters:
            losses = estimate_loss()
            val_log.append((it, losses['val']))
            summ += f" | Vl: {losses['val']:.4f}"

        summ += f" | lr:{lr:.4f} | gn:{gn:.4f}"
        if is_log_it or is_val_it or it == tcfg.max_iters:
            print(f'\r{summ}', flush=True)
        else:
            print(f'\r{summ}', end='', flush=True)

        if it % 2_000 == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'checkpoints/nanocoder_{it}.pt')
            print(f"Saved checkpoint at iter {it}")

    model.eval()
    return model, train_log, val_log, lr_log, gn_log
