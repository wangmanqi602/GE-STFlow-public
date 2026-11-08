"""
TSDiff 基线 (Kollovieh et al., NeurIPS 2023) — 独立复现代码
============================================================
"Predict, Refine, Synthesize: Self-Guiding Diffusion Models for
Probabilistic Time Series Forecasting" 的 SST 适配版:
    - 无条件 DDPM (线性schedule, 1000训练步, ε-预测) 直接在 60 天轨迹上训练
    - 推理时观测自引导 (observation self-guidance): 以目标点最后一天观测值
      y_last 为已知观测, 每步对 z_t 沿 -∇ MSE(x0_hat[:,0], y_last) 引导 w 步长
输入: 仅用目标点历史最后一天值做引导; 训练时不需要历史条件 (与论文一致)。
指标口径: the reference setting 逐窗口中心点目标, 统一指标函数。种子42。
用法: python tsdiff.py --gpu 0
输出: results_tsdiff/eval_results.json + test_export.pt + best_model.pt
"""

import argparse, json, random
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import build_datasets  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_tsdiff"
SEED = 42
T_OUT = 60
D_MODEL, N_HEADS, N_LAYERS = 128, 8, 4
N_TRAIN_T, N_SAMPLE_T = 1000, 100
GUIDE_W = 0.1   # 自引导步长


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def linear_beta_schedule(T, b0=1e-4, bT=0.02):
    return torch.linspace(b0, bT, T)


class TimeEmb(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        half = d_model // 2
        self.w = nn.Parameter(torch.randn(1, half) * 0.1)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, t):
        e = t[:, None] * self.w
        return self.mlp(torch.cat([e.sin(), e.cos()], 1))


class Denoiser(nn.Module):
    """Transformer去噪器 ε_θ(z_t, t), 无条件 (TSDiff 原文)."""

    def __init__(self, d_model=128, n_heads=8, n_layers=4):
        super().__init__()
        self.val_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, T_OUT, d_model) * 0.02)
        self.time_emb = TimeEmb(d_model)
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, z_t, t):      # z_t [B,60], t [B]
        x = self.val_proj(z_t.unsqueeze(-1)) + self.pos
        c = self.time_emb(t)        # [B, d]
        x = x + self.attn(self.norm1(x + c.unsqueeze(1)),
                          self.norm1(x + c.unsqueeze(1)),
                          self.norm1(x + c.unsqueeze(1)))[0]
        x = x + self.ffn(self.norm2(x))
        return self.head(x).squeeze(-1)


class TSDiff(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.denoiser = Denoiser(D_MODEL, N_HEADS, N_LAYERS)
        self.register_buffer("betas", linear_beta_schedule(T))
        alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(alphas, 0)
        self.T = T

    def q_sample(self, y0, t, noise):
        ab = self.alpha_bar[t.cpu()].to(y0.device).view(-1, 1)
        return ab.sqrt() * y0 + (1 - ab).sqrt() * noise

    def forward(self, y0):          # 无条件训练
        B = y0.shape[0]
        t = torch.randint(0, self.T, (B,), device=y0.device)
        noise = torch.randn_like(y0)
        z_t = self.q_sample(y0, t, noise)
        eps_hat = self.denoiser(z_t, t.float() / self.T)
        return F.mse_loss(eps_hat, noise)

    def sample(self, B, y_last, n_steps=100, seed=42, guide_w=GUIDE_W, device="cpu"):
        """DDPM采样 + 观测自引导: x0_hat 的第0天被引导向 y_last.
        注意: 自引导需要 autograd, 故不能用 no_grad 装饰器 (去噪器单独包 no_grad)."""
        rng = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(B, T_OUT, generator=rng, device=device)
        y_last = y_last.to(device)
        ts = torch.linspace(self.T - 1, 0, n_steps + 1, device=device).long()
        for i in range(n_steps):
            t, t_next = ts[i], ts[i + 1]
            tb = torch.full((B,), t, device=device).float() / self.T
            ab = self.alpha_bar[t.cpu()].to(device)
            ab_next = self.alpha_bar[t_next.cpu()].to(device) if t_next >= 0 \
                else torch.tensor(1.0, device=device)
            with torch.no_grad():
                eps = self.denoiser(z, tb).detach()
            # 观测自引导 (TSDiff 核心): 沿 -∇_{z} MSE(x0_hat[:,0], y_last) 步进
            z.requires_grad_(True)
            x0_g = (z - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(1e-6)
            loss_g = F.mse_loss(x0_g[:, 0], y_last)
            grad = torch.autograd.grad(loss_g, z)[0]
            z = (z - guide_w * (1 - ab).sqrt() * grad).detach()
            mu = (z - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(1e-6)
            z = ab_next.sqrt() * mu
            if t_next > 0:
                z = z + (1 - ab_next).sqrt() * torch.randn_like(z)
        return z


# ---------- 统一指标计算 ----------
def compute_metrics(pred, tgt, mean, std):
    pred = torch.as_tensor(pred, dtype=torch.float32)
    tgt = torch.as_tensor(tgt, dtype=torch.float32)
    mse = torch.mean((pred - tgt) ** 2).item(); rmse = mse ** 0.5
    mae = torch.mean(torch.abs(pred - tgt)).item()
    pred_raw = pred * std + mean; tgt_raw = tgt * std + mean
    mae_raw = torch.mean(torch.abs(pred_raw - tgt_raw)).item()
    mask = torch.abs(tgt_raw) > 1.0
    mape = (torch.abs((pred_raw[mask] - tgt_raw[mask]) / tgt_raw[mask]) * 100).mean().item() \
        if mask.sum() > 0 else 0.0
    plist = []
    for b in range(pred.shape[0]):
        p, t = pred[b], tgt[b]
        if p.std() > 1e-8 and t.std() > 1e-8:
            plist.append(((p - p.mean()) * (t - t.mean())).mean() / (p.std() * t.std()).item())
    pearson = float(np.mean(plist)) if plist else float("nan")
    ne = pred_raw.numel()
    psum, psq = pred_raw.sum().item(), (pred_raw ** 2).sum().item()
    tsum, tsq = tgt_raw.sum().item(), (tgt_raw ** 2).sum().item()
    sp = (psq / ne - (psum / ne) ** 2) ** 0.5
    so = (tsq / ne - (tsum / ne) ** 2) ** 0.5
    sstot = tsq - tsum ** 2 / ne; ssres = ((pred_raw - tgt_raw) ** 2).sum().item()
    r2 = 1.0 - ssres / sstot if sstot > 0 else float("nan")
    e = (pred_raw - tgt_raw).numpy()
    return {"mse": mse, "rmse": rmse, "mae": mae, "mae_raw(°C)": mae_raw, "mape(%)": mape,
            "pearson_r": pearson, "r2": r2, "sigma_pred(°C)": sp, "sigma_obs(°C)": so,
            "n_samples": pred.shape[0], "rmse_per_lead": (e ** 2).mean(0) ** 0.5 / std,
            "mae_per_lead": np.abs(e).mean(0) / std, "pred": pred, "tgt": tgt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    seed_all(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    arrs, mean, std = build_datasets()
    tgt = lambda sp: torch.from_numpy(arrs[sp]["tgt"]).float()
    last = lambda sp: torch.from_numpy(arrs[sp]["src"][:, -1, 10, 10]).float()
    train_t, val_t, test_t = tgt("train"), tgt("val"), tgt("test")
    test_last = last("test")

    model = TSDiff(N_TRAIN_T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")

    best_val_mae, best_sd = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_t))
        el, nb = 0.0, 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            yb = train_t[idx].to(device)
            opt.zero_grad()
            loss = model(yb)
            loss.backward(); opt.step()
            el += loss.item() * len(idx); nb += len(idx)
        model.eval()
        val_last = arrs["val"]["src"][:, -1, 10, 10]
        pv = torch.cat([model.sample(len(val_t[i:i + 256]),
                                     torch.from_numpy(val_last[i:i + 256]).float(),
                                     n_steps=16, seed=SEED, device=device).cpu()
                        for i in range(0, len(val_t), 256)])
        val_mae = F.l1_loss(pv, val_t).item()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}/{args.epochs} | loss={el / nb:.5f} | val_mae={val_mae:.5f}", flush=True)

    model.load_state_dict(best_sd)
    torch.save({"model_state_dict": best_sd}, OUT / "best_model.pt")
    last_arr = arrs["test"]["src"][:, -1, 10, 10]
    pt = torch.cat([model.sample(len(test_t[i:i + 256]),
                                 torch.from_numpy(last_arr[i:i + 256]).float(),
                                 n_steps=100, seed=SEED, device=device).cpu()
                    for i in range(0, len(test_t), 256)])
    n = len(pt) // 64 * 64
    pt, tt = pt[:n], test_t[:n]
    m = compute_metrics(pt, tt, mean, std)
    print("=" * 72)
    print(f"TSDiff | RMSE={m['rmse']:.6f} | MAE_raw={m['mae_raw(°C)']:.4f}°C | MAPE={m['mape(%)']:.2f}% | "
          f"Pearson={m['pearson_r']:.4f} | R²={m['r2']:.4f} | σ_pred={m['sigma_pred(°C)']:.4f} | N={m['n_samples']}")
    print("=" * 72)
    summary = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in m.items() if k not in ("pred", "tgt")}
    json.dump(summary, open(OUT / "eval_results.json", "w"), indent=2)
    torch.save({"pred": m["pred"], "tgt": m["tgt"], "mean": mean, "std": std,
                "error": (m["pred"] * std + mean - m["tgt"] * std + mean).numpy(),
                "rmse_per_lead": m["rmse_per_lead"], "mae_per_lead": m["mae_per_lead"]},
               OUT / "test_export.pt", _use_new_zipfile_serialization=False)
    print(f"已保存: {OUT}")


if __name__ == "__main__":
    main()
