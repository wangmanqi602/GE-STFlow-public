"""
Persistent (持续预测) 基线 — 与 the reference setting 完全同口径
=================================================
预测 = 目标点最近观测值 (或最近7日均值), 与 the reference setting 同一测试集样本:
    区域 15-25°N/165-175°E, 121 个 5°×5° 子窗口, T_in=30, T_out=60
    2025 测试, drop_last=True batch_size=64 (与 the reference setting dataloader.py 一致)
指标与 the reference setting engine.autoregressive_test_one_epoch 一致:
    MSE/RMSE: 标准化空间全局平均;  MAE_raw: 原始°C;  MAPE: 跳过|真值|<1°C
    R²: 原始°C;  逐lead RMSE/MAE: 原始°C
"""

import json
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
BATCH = 64  # 与 the reference setting dataloader 一致


def main():
    d = np.load(CACHE / "samples_test.npz")
    m = np.load(CACHE / "samples_meta.npz")
    src, tgt = d["src"], d["tgt"]          # 标准化空间 [N, 30, 20, 20] / [N, 60]
    mean, std = float(m["mean"]), float(m["std"])

    # 与 the reference setting 相同的 drop_last: 取前 N//64*64 个样本
    n = (len(src) // BATCH) * BATCH
    src, tgt = src[:n], tgt[:n]
    i_c, j_c = 10, 10                      # 目标点 = 窗口中心

    variants = {
        "persist_1d": np.repeat(src[:, -1:, i_c, j_c], 60, axis=1),   # 最近一天
        "persist_7d": np.repeat(src[:, -7:, i_c, j_c].mean(1, keepdims=True), 60, axis=1),  # 最近7日均值
    }

    results = {}
    for name, pred in variants.items():
        mse = np.mean((pred - tgt) ** 2)
        rmse = mse ** 0.5
        mae = np.mean(np.abs(pred - tgt))
        pred_raw = pred * std + mean
        tgt_raw = tgt * std + mean
        mae_raw = np.mean(np.abs(pred_raw - tgt_raw))
        mask = np.abs(tgt_raw) > 1.0
        mape = np.mean(np.abs((pred_raw[mask] - tgt_raw[mask]) / tgt_raw[mask]) * 100.0)
        err = pred_raw - tgt_raw
        ss_tot = np.sum((tgt_raw - tgt_raw.mean()) ** 2)
        ss_res = np.sum(err ** 2)
        r2 = 1.0 - ss_res / ss_tot
        results[name] = {
            "mse": float(mse), "rmse": float(rmse), "mae": float(mae),
            "mae_raw(°C)": float(mae_raw), "mape(%)": float(mape), "r2": float(r2),
            "rmse_per_lead(°C)": (err ** 2).mean(0) ** 0.5,
            "mae_per_lead(°C)": np.abs(err).mean(0),
            "n_samples": n,
        }
        print(f"{name:>10}: RMSE={rmse:.4f}  MAE={mae:.4f}  MAE={mae_raw:.4f}°C  "
              f"MAPE={mape:.2f}%  R²={r2:.4f}  (N={n})")

    out = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
               for kk, vv in v.items()} for k, v in results.items()}
    with open(CACHE.parent / "persistence_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("已保存 persistence_results.json")


if __name__ == "__main__":
    main()
