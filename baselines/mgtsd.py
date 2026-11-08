"""
MG-TSD 基线 (Fan et al., AAAI 2024) — 独立复现代码
====================================================
"Multi-Granularity Time Series Diffusion Models" 的 SST 适配版:
    - 多粒度分解: 粗粒度 = 4倍下采样的趋势 (60天 -> 15天, 线性插值回60),
      细粒度 = 残差细节 (y - 粗粒度)
    - 两个 DDPM 去噪器 (共享时间嵌入, 独立权重) 分别对粗/细粒度去噪
    - 条件: GRU 编码30天历史 (与原 DiffusionTS 基线一致, 保证条件对齐)
    - 推理: 两粒度分别采样后相加
指标口径: the reference setting 逐窗口中心点目标, 统一指标函数。种子42。
用法: python mgtsd.py --gpu 0
输出: results_mgtsd/eval_results.json + test_export.pt + best_model.pt
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

OUT = Path(__file__).resolve().parent / "results_mgtsd"
SEED = 42
T_IN, T_OUT = 30, 60
D_MODEL, N_HEADS, N_LAYERS = 128, 8, 4
N_TRAIN_T, N_SAMPLE_T = 1000, 100
COARSE_FACTOR = 4   # 粗粒度下采样倍数 (60 -> 15)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def linear_beta_schedule(T, b0=1e-4, bT=0.02):
    return torch.linspace(b0, bT, T)


def coarse_upsample(y, factor):
    """60天 -> 15天平均 -> 插值回60 (粗粒度趋势)."""
    B, L = y.shape
    yp = F.avg_pool1d(y.unsqueeze(1), factor).squeeze(1)      # [B, 15]
    yp = F.interpolate(yp.unsqueeze(1), size=L, mode="linear",
                       align_corners=True).squeeze(1)          # [B, 60]
    return yp


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


class GRUCond(nn.Module):
    def __init__(self, d_model=128, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(1, d_model, n_layers, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.proj(out[:, -1])


class Denoiser(nn.Module):
    """条件Transformer去噪器 ε_θ(z_t, t, cond): 自注意力 + 时间/GRU条件."""

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

    def forward(self, z_t, t, cond):    # z_t [B,60], t [B], cond [B,d]
        x = self.val_proj(z_t.unsqueeze(-1)) + self.pos
        c = self.time_emb(t).unsqueeze(1) + cond.unsqueeze(1)
        x = x + self.attn(self.norm1(x + c), self.norm1(x + c), self.norm1(x + c))[0]
        x = x + self.ffn(self.norm2(x))
        return self.head(x).squeeze(-1)


class MGTSD(nn.Module):
    def __init__(self, T=1000, factor=4):
        super().__init__()
        self.factor = factor
        self.cond = GRUCond(D_MODEL)
        self.denoiser_coarse = Denoiser(D_MODEL, N_HEADS, N_LAYERS)
        self.denoiser_fine = Denoiser(D_MODEL, N_HEADS, N_LAYERS)
        self.register_buffer("betas", linear_beta_schedule(T))
        alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(alphas, 0)
        self.T = T

    def q_sample(self, y0, t, noise):
        ab = self.alpha_bar[t.cpu()].to(y0.device).view(-1, 1)
        return ab.sqrt() * y0 + (1 - ab).sqrt() * noise

    def forward(self, h, y0):       # h [B,30], y0 [B,60]
        B = h.shape[0]
        cond = self.cond(h)
        t = torch.randint(0, self.T, (B,), device=h.device)
        # 粗粒度 (趋势) + 细粒度 (残差)
        yc = coarse_upsample(y0, self.factor)
        yf = y0 - yc
        loss = 0.0
        for y, dn in ((yc, self.denoiser_coarse), (yf, self.denoiser_fine)):
            noise = torch.randn_like(y)
            z_t = self.q_sample(y, t, noise)
            eps_hat = dn(z_t, t.float() / self.T, cond)
            loss = loss + F.mse_loss(eps_hat, noise)
        return loss

    @torch.no_grad()
    def sample(self, h, n_steps=100, seed=42, device="cpu"):
        B = h.shape[0]
        cond = self.cond(h)
        rng = torch.Generator(device=device).manual_seed(seed)
        out = torch.zeros(B, T_OUT, device=device)
        for init, dn in ((None, self.denoiser_coarse), (None, self.denoiser_fine)):
            z = torch.randn(B, T_OUT, generator=rng, device=device)
            ts = torch.linspace(self.T - 1, 0, n_steps + 1, device=device).long()
            for i in range(n_steps):
                t, t_next = ts[i], ts[i + 1]
                tb = torch.full((B,), t, device=device).float() / self.T
                ab = self.alpha_bar[t.cpu()].to(device)
                ab_next = self.alpha_bar[t_next.cpu()].to(device) if t_next >= 0 \
                    else torch.tensor(1.0, device=device)
                eps = dn(z, tb, cond)
                mu = (z - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(1e-6)
                z = ab_next.sqrt() * mu
                if t_next > 0:
                    z = z + (1 - ab_next).sqrt() * torch.randn_like(z)
            # 粗粒度结果插值回60后加到输出
            if init is None:
                out = out + z
            else:
                out = out + z
        # 两粒度简单相加 (粗粒度已在60维空间去噪, 无需插值)
        return out


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
    h = lambda sp: torch.from_numpy(arrs[sp]["src"][:, :, 10, 10]).float()
    tgt = lambda sp: torch.from_numpy(arrs[sp]["tgt"]).float()
    train_h, val_h, test_h = h("train"), h("val"), h("test")
    train_t, val_t, test_t = tgt("train"), tgt("val"), tgt("test")

    model = MGTSD(N_TRAIN_T, COARSE_FACTOR).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")

    best_val_mae, best_sd = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_h))
        el, nb = 0.0, 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            xb, yb = train_h[idx].to(device), train_t[idx].to(device)
            opt.zero_grad()
            loss = model(xb, yb)
            loss.backward(); opt.step()
            el += loss.item() * len(idx); nb += len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model.sample(val_h[i:i + 256].to(device),
                                         n_steps=16, seed=SEED, device=device).cpu()
                            for i in range(0, len(val_h), 256)])
        val_mae = F.l1_loss(pv, val_t).item()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}/{args.epochs} | loss={el / nb:.5f} | val_mae={val_mae:.5f}", flush=True)

    model.load_state_dict(best_sd)
    torch.save({"model_state_dict": best_sd}, OUT / "best_model.pt")
    with torch.no_grad():
        pt = torch.cat([model.sample(test_h[i:i + 256].to(device),
                                     n_steps=100, seed=SEED, device=device).cpu()
                        for i in range(0, len(test_h), 256)])
        n = len(pt) // 64 * 64
        pt, tt = pt[:n], test_t[:n]
    m = compute_metrics(pt, tt, mean, std)
    print("=" * 72)
    print(f"MG-TSD | RMSE={m['rmse']:.6f} | MAE_raw={m['mae_raw(°C)']:.4f}°C | MAPE={m['mape(%)']:.2f}% | "
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
