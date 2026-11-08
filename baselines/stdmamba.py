"""
STDMamba 基线 (IEEE 2025, SST专用) — 独立复现代码
==================================================
"Spatiotemporal Decomposition Mamba for Long-Term Fine-Grained SST Prediction"
的单变量适配版 (纯 PyTorch 实现 S6 选择性状态空间):
    - 高斯加权序列分解: 趋势 (高斯核移动平均) + 细节 (残差)
    - 双 Mamba(S6) 分支分别编码趋势与细节, 末状态拼接 -> Linear(2d->60)
    - 论文核心思想: 分解 + 状态空间长程建模
指标口径: the reference setting 逐窗口中心点目标, 统一指标函数, 种子42。
用法: python stdmamba.py --gpu 0
输出: results_stdmamba/eval_results.json + test_export.pt + best_model.pt
"""

import argparse, json, math, random
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import build_datasets  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_stdmamba"
SEED = 42
T_IN, T_OUT = 30, 60
D_MODEL, D_STATE = 128, 16


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class GaussianDecomp(nn.Module):
    """高斯加权序列分解 (STDMamba 论文): 趋势 = 高斯核滑动平均, 细节 = x - 趋势."""

    def __init__(self, kernel=15, sigma=2.0):
        super().__init__()
        self.kernel = kernel
        half = kernel // 2
        g = torch.exp(-0.5 * (torch.arange(-half, half + 1) / sigma) ** 2)
        self.register_buffer("w", g / g.sum())

    def forward(self, x):       # [B, L]
        pad = F.pad(x, (self.kernel // 2, self.kernel // 2), mode="reflect")
        trend = F.conv1d(pad.unsqueeze(1), self.w.view(1, 1, -1)).squeeze(1)
        return trend, x - trend


class S6Block(nn.Module):
    """选择性状态空间模型 S6 (Mamba 核心), 纯 PyTorch 实现.
    x [B,L,D] -> y [B,L,D] (因果卷积 + 选择扫描 + 门控)."""

    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, groups=d_model,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1)
        self.dt_proj = nn.Linear(2 * d_state + 1, d_model)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):       # [B, L, D]
        B, L, D = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)             # [B,L,D] x2
        u = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)
        BC = self.x_proj(u)                                  # [B,L,2S+1]
        Bc, C, delta = BC.split([self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt_proj(BC))                 # [B,L,D]
        A = -torch.exp(self.A_log)                           # [D,S]
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outs = []
        for t in range(L):
            dA = torch.exp(delta[:, t].unsqueeze(-1) * A)    # [B,D,S]
            h = dA * h + Bc[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            y = (C[:, t].unsqueeze(1) * h).sum(-1) + self.D * u[:, t]
            outs.append(y)
        y = torch.stack(outs, dim=1)                         # [B,L,D]
        return self.out_proj(y * F.silu(z)) + x


class STDMamba(nn.Module):
    def __init__(self, seq_len=30, pred_len=60, d_model=128, d_state=16, n_layers=1):
        super().__init__()
        self.decomp = GaussianDecomp(kernel=15)
        self.emb_t = nn.Linear(1, d_model)
        self.emb_d = nn.Linear(1, d_model)
        self.mamba_t = nn.ModuleList([S6Block(d_model, d_state) for _ in range(n_layers)])
        self.mamba_d = nn.ModuleList([S6Block(d_model, d_state) for _ in range(n_layers)])
        self.head = nn.Linear(2 * d_model, pred_len)

    def forward(self, x):       # [B, 30]
        trend, detail = self.decomp(x)
        xt = self.emb_t(trend.unsqueeze(-1))
        xd = self.emb_d(detail.unsqueeze(-1))
        for blk in self.mamba_t:
            xt = blk(xt)
        for blk in self.mamba_d:
            xd = blk(xd)
        z = torch.cat([xt[:, -1], xd[:, -1]], dim=-1)   # 两分支末状态
        return self.head(z)


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

    model = STDMamba(T_IN, T_OUT, D_MODEL, D_STATE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            loss = F.mse_loss(model(xb), yb)
            loss.backward(); opt.step()
            el += loss.item() * len(idx); nb += len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(val_h[i:i + 512].to(device)).cpu()
                            for i in range(0, len(val_h), 512)])
        val_mae = F.l1_loss(pv, val_t).item()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}/{args.epochs} | loss={el / nb:.5f} | val_mae={val_mae:.5f}", flush=True)

    model.load_state_dict(best_sd)
    torch.save({"model_state_dict": best_sd}, OUT / "best_model.pt")
    with torch.no_grad():
        pt = torch.cat([model(test_h[i:i + 512].to(device)).cpu()
                        for i in range(0, len(test_h), 512)])
        n = len(pt) // 64 * 64
        pt, tt = pt[:n], test_t[:n]
    m = compute_metrics(pt, tt, mean, std)
    print("=" * 72)
    print(f"STDMamba | RMSE={m['rmse']:.6f} | MAE_raw={m['mae_raw(°C)']:.4f}°C | MAPE={m['mape(%)']:.2f}% | "
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
