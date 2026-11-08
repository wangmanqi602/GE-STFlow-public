"""
气候态基线 (Climatology) — 独立复现代码
=======================================
定义: 对每个子窗口中心点, 用训练年份(2022-2023)逐日平均SST (按年积日DOY)
     作为该日历日的气候态; 测试样本的60天预报 = 目标期60个日历日的气候态序列.
数据口径: 与 the reference setting 完全一致 (同区域/同窗口/同时间划分/同标准化).
指标口径: 与 evaluate.py 一致 (RMSE/MAE/MAPE/R²/Pearson, 标准化空间 + 原始°C).

用法: python climatology.py   (CPU即可, 无需训练)
输出: results_climatology/eval_results.json + test_export.pt
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# 复用项目数据管线 (与所有基线同一数据源)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import build_datasets, make_loaders, CONFIG  # noqa: E402
from data_extract import open_sst_dataset, split_by_time, generate_sub_windows  # noqa: E402
from config import CONFIG as _C  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_climatology"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------- 统一指标计算 (与所有基线脚本一致) ----------
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
    d = CONFIG.data
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. 用原始SST构建 每格点×每DOY 的气候态 (仅训练年份2022-2023) ----
    ds, sst = open_sst_dataset(d.data_path, variable_name=d.variable_name,
                               lat_range=d.lat_value_range, lon_range=d.lon_value_range)
    train_sst, _, _ = split_by_time(sst, train_time=d.train_time,
                                    val_time=d.val_time, test_time=d.test_time)
    lat_vals = sst["lat"].values.astype("float32")
    lon_vals = sst["lon"].values.astype("float32")
    windows = generate_sub_windows(lat_values=lat_vals, lon_values=lon_vals,
                                   lat_span=d.sub_lat_span, lon_span=d.sub_lon_span,
                                   stride_lat=d.stride_lat, stride_lon=d.stride_lon)
    doys = pd.to_datetime(train_sst.time.values).dayofyear.values - 1   # 0-364
    raw = train_sst.values  # [T, 40, 40], 可能含NaN
    clim = np.full((365, raw.shape[1], raw.shape[2]), np.nan, dtype=np.float64)
    for doy in range(365):
        m = (doys == doy)
        if m.sum() > 0:
            with np.errstate(all="ignore"):
                clim[doy] = np.nanmean(raw[m], axis=0)
    print(f"气候态: 训练年份 {np.unique(doys // 365 + 2022)} 逐DOY平均, shape={clim.shape}")

    # ---- 2. 测试集预报: 目标60天取对应DOY气候态 (窗口中心点) ----
    arrs, mean, std = build_datasets()
    _, _, test_loader = make_loaders(arrs, 64, num_workers=4, drop_last_test=True)
    from data_extract import make_start_indices
    test_sst, = split_by_time(sst, train_time=d.train_time, val_time=d.val_time,
                              test_time=d.test_time)[2:3]
    starts = make_start_indices(test_sst, d.t_in, d.t_out)
    # 标准化统计
    preds, tgts = [], []
    t0 = time.time()
    for i, batch in enumerate(test_loader):
        b = len(batch["target_trajectory"])
        base = i * 64  # drop_last=True, 严格按 batch 顺序
        batch_pred = np.empty((b, 60), dtype=np.float32)
        for j in range(b):
            si = base + j
            win = si % len(windows)
            start = si // len(windows)
            start_doy = pd.to_datetime(test_sst.time.values[start]).dayofyear - 1
            w = windows[win]
            ic, jc = w["lat_start"] + w["target_lat_idx"], w["lon_start"] + w["target_lon_idx"]
            target_doys = (start_doy + d.t_in + np.arange(60)) % 365
            c = np.stack([clim[doy, ic, jc] for doy in target_doys])
            c = np.where(np.isnan(c), mean, c)              # NaN -> 训练均值 (标准化后为0)
            batch_pred[j] = (c - mean) / std                # 转标准化空间
        preds.append(batch_pred)
        tgts.append(batch["target_trajectory"].numpy())
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test_loader)} ({time.time() - t0:.0f}s)", flush=True)
    pred = np.concatenate(preds)
    tgt = np.concatenate(tgts)

    # ---- 3. 统一指标 ----
    m = compute_metrics(pred, tgt, mean, std)
    print("\n" + "=" * 72)
    print(f"Climatology | RMSE={m['rmse']:.6f} | MAE_raw={m['mae_raw(°C)']:.4f}°C | "
          f"MAPE={m['mape(%)']:.2f}% | Pearson={m['pearson_r']:.4f} | R²={m['r2']:.4f} | "
          f"σ_pred={m['sigma_pred(°C)']:.4f} | σ_obs={m['sigma_obs(°C)']:.4f} | N={m['n_samples']}")
    print("=" * 72)

    summary = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in m.items() if k not in ("pred", "tgt")}
    with open(OUT / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"pred": m["pred"], "tgt": m["tgt"], "mean": mean, "std": std,
                "error": (m["pred"] * std + mean - m["tgt"] * std + mean).numpy(),
                "rmse_per_lead": m["rmse_per_lead"], "mae_per_lead": m["mae_per_lead"]},
               OUT / "test_export.pt", _use_new_zipfile_serialization=False)
    print(f"已保存: {OUT / 'eval_results.json'} + test_export.pt")


if __name__ == "__main__":
    main()
