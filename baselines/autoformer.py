"""
Autoformer 基线 (Wu et al., NeurIPS 2021) — 独立复现代码
=========================================================
输入: 目标点30天SST历史 (标准化) [B,30] -> 输出: 未来60天 [B,60]
架构: 序列分解(移动平均kernel=25) + Auto-Correlation(FFT时延聚合, top-k)
      编码器2层 + 解码器1层, d_model=128, 4头.
训练: Adam lr=1e-3, MSE, 30 epochs, 按验证集MAE选最优, 固定种子42.
指标口径: 与 evaluate.py 一致 (RMSE/MAE/MAPE/R²/Pearson).

用法: python autoformer.py --gpu 0
输出: results_autoformer/eval_results.json + test_export.pt + best_model.pt
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

OUT = Path(__file__).resolve().parent / "results_autoformer"
SEED = 42
T_IN, T_OUT = 30, 60
D_MODEL, N_HEADS = 128, 4


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):   # [B, L, D]
        k = self.kernel
        pad = F.pad(x, (0, 0, k // 2, k // 2), mode="reflect")
        trend = F.avg_pool1d(pad.permute(0, 2, 1), k, 1).permute(0, 2, 1)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """FFT自相关 + 时延聚合 (论文公式4-5)."""

    def __init__(self, d_model, n_heads=4, factor=1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def time_delay_agg(self, v, corr):
        # v [B, L, H, D_k], corr [B, L, H] -> [B, L, H, D_k]
        B, L, H, D = v.shape
        top_k = max(1, int(self.factor * math.log(L)))
        weights, delay = torch.topk(corr, top_k, dim=1)     # [B, top_k, H]
        weights = F.softmax(weights, dim=1)                 # 归一化聚合权重
        out = torch.zeros_like(v)
        idx_base = torch.arange(L, device=v.device)
        for i in range(top_k):
            d = delay[:, i, :]                              # [B, H]
            # 目标索引 idx[b,l,h] = (l - d[b,h]) mod L
            idx = (idx_base[None, :, None] - d[:, None, :]).remainder(L)  # [B, L, H]
            gathered = v.gather(1, idx.unsqueeze(-1).expand(B, L, H, D))
            out += weights[:, i, :].view(B, 1, H, 1) * gathered
        return out

    def forward(self, q, k, v):
        # q,k,v [B, L_q, D], k/v 长度可能不同 (交叉注意力) -> 补零对齐 (官方实现同款)
        B, L, D = q.shape
        S = k.shape[1]
        q = self.q(q).view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.k(k).view(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v(v).view(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        if L > S:
            z = torch.zeros(B, L - S, self.n_heads, self.d_k, device=q.device)
            k = torch.cat([k, z.permute(0, 2, 1, 3)], dim=2)
            v = torch.cat([v, z.permute(0, 2, 1, 3)], dim=2)
        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        S = torch.fft.irfft(q_fft * k_fft.conj(), n=L, dim=2).mean(-1)  # [B,H,L]
        corr = S.permute(0, 2, 1)                                       # [B,L,H]
        v2 = v.permute(0, 2, 1, 3)                                      # [B,L,H,D_k]
        out = self.time_delay_agg(v2, corr)                             # [B,L,H,D_k]
        out = out.permute(0, 1, 2, 3).reshape(B, L, D)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, kernel=25):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel)
        self.decomp2 = SeriesDecomp(kernel)
        self.ac = AutoCorrelation(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x, _ = self.decomp1(x + self.ac(x, x, x))
        x, _ = self.decomp2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, kernel=25):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel)
        self.decomp2 = SeriesDecomp(kernel)
        self.decomp3 = SeriesDecomp(kernel)
        self.ac1 = AutoCorrelation(d_model, n_heads)
        self.ac2 = AutoCorrelation(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x, enc_out):
        x, _ = self.decomp1(x + self.ac1(x, x, x))
        x, _ = self.decomp2(x + self.ac2(x, enc_out, enc_out))
        x, _ = self.decomp3(x + self.ffn(x))
        return x


class Autoformer(nn.Module):
    def __init__(self, seq_len=30, pred_len=60, d_model=128, n_heads=4,
                 enc_layers=2, dec_layers=1, kernel=25):
        super().__init__()
        # 解码器输入 = 输入后半段(seq_len//2) + 全预测长度(pred_len)
        self.pos = nn.Parameter(torch.randn(1, seq_len // 2 + pred_len, d_model) * 0.02)
        self.enc_embed = nn.Linear(1, d_model)
        self.dec_embed = nn.Linear(1, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, kernel)
                                      for _ in range(enc_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, kernel)
                                      for _ in range(dec_layers)])
        self.head = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, x):   # [B, 30]
        B, L = x.shape
        enc = self.enc_embed(x.unsqueeze(-1)) + self.pos[:, :L]
        for layer in self.encoder:
            enc = layer(enc)
        # 解码器初始: 输入后半段 + 占位0 (论文标准)
        dec_in = torch.cat([x[:, -L // 2:], torch.zeros(B, self.pred_len, device=x.device)], 1)
        dec = self.dec_embed(dec_in.unsqueeze(-1)) + self.pos
        for layer in self.decoder:
            dec = layer(dec, enc)
        return self.head(dec).squeeze(-1)[:, -self.pred_len:]


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
            "n_samples": pred.shape[0], "rmse_per_lead": (e ** 2).mean(0) ** 0.5,
            "mae_per_lead": np.abs(e).mean(0), "pred": pred, "tgt": tgt}


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

    model = Autoformer(T_IN, T_OUT, D_MODEL, N_HEADS).to(device)
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
    print(f"Autoformer | RMSE={m['rmse']:.6f} | MAE_raw={m['mae_raw(°C)']:.4f}°C | MAPE={m['mape(%)']:.2f}% | "
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
