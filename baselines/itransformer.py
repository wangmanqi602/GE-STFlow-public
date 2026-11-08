"""
iTransformer 基线 (Liu et al., ICLR 2024) — 独立复现代码
=========================================================
输入: 目标点30天SST历史 (标准化) [B,30] -> 输出: 未来60天 [B,60]
单变量适配: 将每个时间步作为 token (inverted embedding 思想),
自注意力跨 30 个时间步 token, 官方式 per-variate 投影 Linear(d->60) 后
对 token 维取平均 -> [B,60]。
架构: 值->d_model 嵌入 + 位置编码, 3层Transformer编码器(d=128, 4头, dropout=0.1)。
训练: AdamW lr=1e-3, MSE, 30 epochs, 按验证集MAE选最优, 固定种子42。
指标口径: the reference setting 逐窗口中心点目标, 统一指标函数 (RMSE/MAE标准化/R²)。

用法: python itransformer.py --gpu 0
输出: results_itransformer/eval_results.json + test_export.pt + best_model.pt
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

OUT = Path(__file__).resolve().parent / "results_itransformer"
SEED = 42
T_IN, T_OUT = 30, 60


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class iTransformer(nn.Module):
    """inverted embedding: 时间步作为 token, 注意力跨 token, FFN 在嵌入维."""

    def __init__(self, seq_len=30, pred_len=60, d_model=128, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)            # 值 -> token 嵌入
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads,
                                           dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, pred_len)      # 官方 per-variate 投影

    def forward(self, x):                       # [B, 30]
        x = self.embed(x.unsqueeze(-1)) + self.pos   # [B, 30, d]
        x = self.encoder(x)
        x = self.proj(x).mean(dim=1)            # [B, 30, 60] -> [B, 60]
        return x


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

    model = iTransformer(T_IN, T_OUT).to(device)
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
            pv = torch.cat([model(val_h[i:i + 1024].to(device)).cpu()
                            for i in range(0, len(val_h), 1024)])
        val_mae = F.l1_loss(pv, val_t).item()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}/{args.epochs} | loss={el / nb:.5f} | val_mae={val_mae:.5f}", flush=True)

    model.load_state_dict(best_sd)
    torch.save({"model_state_dict": best_sd}, OUT / "best_model.pt")
    with torch.no_grad():
        pt = torch.cat([model(test_h[i:i + 1024].to(device)).cpu()
                        for i in range(0, len(test_h), 1024)])
        n = len(pt) // 64 * 64
        pt, tt = pt[:n], test_t[:n]
    m = compute_metrics(pt, tt, mean, std)
    print("=" * 72)
    print(f"iTransformer | RMSE={m['rmse']:.6f} | MAE_raw={m['mae_raw(°C)']:.4f}°C | MAPE={m['mape(%)']:.2f}% | "
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
