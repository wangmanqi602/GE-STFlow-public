"""GE-STFlow 训练: 值空间扩散"""
import argparse, json, os, random, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from dataset import build_datasets, make_loaders
from models.gestflow import GESTFlow


@dataclass
class Config:
    T_in: int = 30; T_out: int = 60
    d_model: int = 256; n_heads: int = 8
    n_encoder_layers: int = 4; n_denoiser_layers: int = 4
    n_time_tokens: int = 4
    noise_scale: float = 1.0; p_mean: float = 0.0; p_std: float = 0.8
    t_eps: float = 5e-2
    self_cond_prob: float = 0.5
    value_loss_weight: float = 1.0    # 值空间L1权重
    use_residual: bool = False
    batch_size: int = 64; blr: float = 0.001
    epochs: int = 30; warmup_epochs: float = 0.5
    weight_decay: float = 0.0; grad_clip: float = 1.0
    ema_decay: float = 0.9999; seed: int = 42
    num_workers: int = 4
    val_batches: int = 8; val_sample_steps: int = 8
    num_sampling_steps: int = 32


def add_noise(e_0, noise, t, noise_scale):
    return t.view(-1, 1) * e_0 + (1 - t.view(-1, 1)) * noise * noise_scale


def sample_t(rng, B, p_mean, p_std, device="cpu"):
    z = torch.randn(B, generator=rng, device=device) * p_std + p_mean
    return torch.sigmoid(z)


def net_out_to_v(x_pred, z_t, t, t_eps):
    return (x_pred - z_t) / (1 - t).clamp(min=t_eps).view(-1, 1)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def apply_to(self, model):
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in self.shadow.items()})


def train_step(model, batch, cfg, rng, opt):
    model.train(); opt.zero_grad()
    device = next(model.parameters()).device
    B = batch["sst_fields"].shape[0]

    memory, last_sst, doy_start = model.encode_memory(
        batch["sst_fields"], batch["coords"],
        batch["target_idx"], batch["land_mask"], batch["doys"],
    )
    target = batch["target_trajectory"]  # [B, T_out]
    if cfg.use_residual and last_sst is not None:
        target = target - last_sst.unsqueeze(-1)

    t = sample_t(rng, B, cfg.p_mean, cfg.p_std, device)
    noise = torch.randn_like(target)
    z_t = add_noise(target, noise, t, cfg.noise_scale)     # [B, T_out]

    use_sc = cfg.self_cond_prob > 0 and torch.rand(1, generator=rng, device=device).item() < cfg.self_cond_prob
    if use_sc:
        with torch.no_grad():
            sc = model.denoise(z_t, t, memory, doy_start, None)
        x_pred = model.denoise(z_t, t, memory, doy_start, sc)
    else:
        x_pred = model.denoise(z_t, t, memory, doy_start, None)

    v_pred = net_out_to_v(x_pred, z_t, t, cfg.t_eps)
    v_target = net_out_to_v(target, z_t, t, cfg.t_eps)
    loss_vel = F.mse_loss(v_pred, v_target)
    loss_val = F.l1_loss(x_pred, target)
    loss = loss_vel + cfg.value_loss_weight * loss_val
    loss.backward()
    if cfg.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    return {"loss": loss.item(), "loss_vel": loss_vel.item(), "loss_val": loss_val.item()}


@torch.no_grad()
def validate(model, loader, cfg, device, max_batches=8, num_steps=8):
    model.eval()
    mae_sum, rmse_sum, n = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = model.sample(
            batch["sst_fields"], batch["coords"], batch["target_idx"],
            batch["land_mask"], batch["doys"],
            num_steps=num_steps, p_mean=cfg.p_mean,
        )
        mae_sum += F.l1_loss(pred, batch["target_trajectory"]).item() * len(pred)
        rmse_sum += (F.mse_loss(pred, batch["target_trajectory"]).item() ** 0.5) * len(pred)
        n += len(pred)
    return {"val_mae": mae_sum / max(n, 1), "val_rmse": rmse_sum / max(n, 1)}


def run_training(cfg, out_dir):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    arrs, mean, std = build_datasets()
    train_loader, val_loader, _ = make_loaders(arrs, cfg.batch_size, num_workers=cfg.num_workers)
    steps_per_epoch = len(train_loader)
    print(f"train batches/epoch: {steps_per_epoch}, val: {len(val_loader)}")

    model = GESTFlow(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers, n_denoiser_layers=cfg.n_denoiser_layers,
        T_in=cfg.T_in, T_out=cfg.T_out, n_time_tokens=cfg.n_time_tokens,
        noise_scale=cfg.noise_scale, t_eps=cfg.t_eps,
        use_residual=cfg.use_residual,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    lr = cfg.blr * cfg.batch_size / 256.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    def lr_f(step):
        return (step + 1) / max(warmup_steps, 1) if step < warmup_steps else 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_f)
    ema = EMA(model, cfg.ema_decay)

    rng = torch.Generator(device=device).manual_seed(cfg.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf"); history = []; global_step = 0
    log_f = open(out_dir / "train.log", "a", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"GE-STFlow start={time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"config: {json.dumps(cfg.__dict__, indent=2)}")

    t_start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        el, ev, ev2, nb = 0.0, 0.0, 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            m = train_step(model, batch, cfg, rng, opt)
            sched.step(); ema.update(model)
            el += m["loss"]; ev += m["loss_vel"]; ev2 += m["loss_val"]; nb += 1
            global_step += 1

        ema.apply_to(model)
        vs = validate(model, val_loader, cfg, device, max_batches=cfg.val_batches, num_steps=cfg.val_sample_steps)
        log(f"epoch {epoch+1}/{cfg.epochs} | loss={el/nb:.5f} vel={ev/nb:.5f} val={ev2/nb:.5f} | "
            f"val_mae={vs['val_mae']:.5f} val_rmse={vs['val_rmse']:.5f} | "
            f"lr={sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s/epoch | step={global_step}")
        history.append({"epoch": epoch+1, "loss": el/nb, "loss_vel": ev/nb, "loss_val": ev2/nb,
                        "val_mae": vs["val_mae"], "val_rmse": vs["val_rmse"]})
        if vs["val_mae"] < best_val_mae:
            best_val_mae = vs["val_mae"]
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.shadow, "config": cfg.__dict__,
                        "val_mae": vs["val_mae"], "history": history}, out_dir / "best_model.pt")
            log(f"  -> saved best (val_mae={best_val_mae:.5f})")

    log(f"done, {(time.time()-t_start)/60:.1f}min, best_val_mae={best_val_mae:.5f}")
    with open(out_dir / "history.json", "w") as f:
        json.dump({"config": cfg.__dict__, "history": history, "best_val_mae": best_val_mae,
                   "mean": mean, "std": std}, f, indent=2)
    log_f.close()
    return out_dir / "best_model.pt"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=256); p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_enc", type=int, default=4); p.add_argument("--n_dns", type=int, default=4)
    p.add_argument("--blr", type=float, default=0.001)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--p_mean", type=float, default=0.0)
    p.add_argument("--value_loss_weight", type=float, default=1.0)
    p.add_argument("--use_residual", action="store_true")
    p.add_argument("--self_cond_prob", type=float, default=0.5)
    p.add_argument("--out", type=str, default="checkpoints/stidiff_v2")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = Config(
        epochs=args.epochs, batch_size=args.batch_size, d_model=args.d_model,
        n_heads=args.n_heads, n_encoder_layers=args.n_enc, n_denoiser_layers=args.n_dns,
        blr=args.blr, noise_scale=args.noise_scale, p_mean=args.p_mean,
        value_loss_weight=args.value_loss_weight, use_residual=args.use_residual,
        self_cond_prob=args.self_cond_prob, seed=args.seed,
    )
    run_training(cfg, args.out)
