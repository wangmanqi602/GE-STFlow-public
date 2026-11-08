"""
TimeMixer 基线 (Wang et al., ICLR 2024) — 独立复现代码
=======================================================
"TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
的单变量适配版:
    - 多尺度金字塔: 平均池化 s∈{2,4,8} 得 30/15/7 三个尺度
    - PDM (past-decomposable mixing): 各尺度分解为趋势(移动平均)与季节分量,
      跨尺度拼接 -> MLP 混合 -> 还原 (残差连接), 2 块
    - FMM (future-multipredictor mixing): 各尺度独立预测60天,
      学习尺度权重 softmax 加权求和
指标口径: the reference setting 逐窗口中心点目标, 统一指标函数, 种子42。
用法: python timemixer.py --gpu 0
输出: results_timemixer/eval_results.json + test_export.pt + best_model.pt
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

OUT = Path(__file__).resolve().parent / "results_timemixer"
SEED = 42
T_IN, T_OUT = 30, 60
SCALES = [2, 4, 8]
PDM_BLOCKS = 2


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def moving_avg(x, kernel):
    pad = F.pad(x, (kernel // 2, kernel // 2), mode="reflect")
    return F.avg_pool1d(pad.unsqueeze(1), kernel, 1).squeeze(1)


class PDMBlock(nn.Module):
    """past-decomposable mixing: 跨尺度混合趋势与季节分量."""

    def __init__(self, lens, kernel=5, d_mix=64):
        super().__init__()
        self.lens = lens
        self.kernel = kernel
        total = sum(lens)
        # 季节/趋势各自的跨尺度混合 MLP
        self.mix_season = nn.Sequential(nn.Linear(total, d_mix), nn.GELU(),
                                        nn.Linear(d_mix, total))
        self.mix_trend = nn.Sequential(nn.Linear(total, d_mix), nn.GELU(),
                                       nn.Linear(d_mix, total))

    def forward(self, xs):
        trends = [moving_avg(x, self.kernel) for x in xs]
        seasons = [x - t for x, t in zip(xs, trends)]
        s_cat = torch.cat(seasons, dim=1)
        t_cat = torch.cat(trends, dim=1)
        s_mix = self.mix_season(s_cat)
        t_mix = self.mix_trend(t_cat)
        outs, off = [], 0
        for L, s, t in zip(self.lens, seasons, trends):
            sm = s_mix[:, off:off + L] + s          # 残差
            tm = t_mix[:, off:off + L] + t
            outs.append(sm + tm)
            off += L
        return outs


class TimeMixer(nn.Module):
    def __init__(self, seq_len=30, pred_len=60, scales=SCALES):
        super().__init__()
        self.scales = scales
        self.lens = [seq_len // s for s in scales]
        self.pdm = nn.ModuleList([PDMBlock(self.lens) for _ in range(PDM_BLOCKS)])
        # FMM: 每尺度一个预测器 + 可学习尺度权重
        self.predictors = nn.ModuleList(
            [nn.Linear(L, pred_len) for L in self.lens])
        self.scale_w = nn.Parameter(torch.zeros(len(scales)))

    def forward(self, x):       # [B, 30]
        xs = [F.avg_pool1d(x.unsqueeze(1), s).squeeze(1) for s in self.scales]
        for blk in self.pdm:
            xs = blk(xs)
        preds = [pred_i(xs[i]) for i, pred_i in enumerate(self.predictors)]
        w = F.softmax(self.scale_w, dim=0)
        out = sum(w[i] * preds[i] for i in range(len(preds)))
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
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    arrs, mean, std = build_datasets()
    h = lambda sp: torch.from_numpy(arrs[sp]["src"][:, :, 10, 10]).float()
    tgt = lambda sp: torch.from_numpy(arrs[sp]["tgt"]).float()
    train_h, val_h, test_h = h("train"), h("val"), h("test")
    train_t, val_t, test_t = tgt("train"), tgt("val"), tgt("test")

    model = TimeMixer(T_IN, T_OUT).to(device)
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
    torch.save({"model_state_dict": best_sd}, out_dir / "best_model.pt")
    with torch.no_grad():
        pt = torch.cat([model(test_h[i:i + 1024].to(device)).cpu()
                        for i in range(0, len(test_h), 1024)])
        n = len(pt) // 64 * 64
        pt, tt = pt[:n], test_t[:n]
    m = compute_metrics(pt, tt, mean, std)
    print("=" * 72)
    print(f"TimeMixer | RMSE={m['rmse']:.6f} | MAE_raw={m['mae_raw(°C)']:.4f}°C | MAPE={m['mape(%)']:.2f}% | "
          f"Pearson={m['pearson_r']:.4f} | R²={m['r2']:.4f} | σ_pred={m['sigma_pred(°C)']:.4f} | N={m['n_samples']}")
    print("=" * 72)
    summary = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in m.items() if k not in ("pred", "tgt")}
    json.dump(summary, open(out_dir / "eval_results.json", "w"), indent=2)
    torch.save({"pred": m["pred"], "tgt": m["tgt"], "mean": mean, "std": std,
                "error": (m["pred"] * std + mean - m["tgt"] * std + mean).numpy(),
                "rmse_per_lead": m["rmse_per_lead"], "mae_per_lead": m["mae_per_lead"]},
               out_dir / "test_export.pt", _use_new_zipfile_serialization=False)
    print(f"已保存: {out_dir}")


if __name__ == "__main__":
    main()
