"""Compute RMSE / MAE / MAPE / R2 / Pearson from a test_export.pt.

Optional seasonal breakdown (DJF/MAM/JJA/SON) using the forecast start day.
"""
import argparse, json
import numpy as np, torch

MONTH_LEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

def month_of_doy(doy):
    cum = np.concatenate([[0], MONTH_LEN.cumsum()])
    return np.clip(np.searchsorted(cum, np.clip(doy, 1, 365), side="right") - 1, 0, 11)

def season_of_month(m):
    return "DJF" if m in (11, 0, 1) else "MAM" if m in (2, 3, 4) else "JJA" if m in (5, 6, 7) else "SON"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--cache", default="data_cache/samples_test.npz")
    args = ap.parse_args()
    e = torch.load(args.export, map_location="cpu", weights_only=False)
    p = torch.as_tensor(e["pred"], dtype=torch.float32)
    t = torch.as_tensor(e["tgt"], dtype=torch.float32)
    mean, std = float(e["mean"]), float(e["std"])
    pr, tr = p * std + mean, t * std + mean
    rmse = float(np.sqrt(((p - t) ** 2).mean()))
    mae = float((p - t).abs().mean())
    mae_raw = float((pr - tr).abs().mean())
    mask = tr.abs() > 1.0
    mape = float(((pr[mask] - tr[mask]).abs() / tr[mask].abs()).mean() * 100)
    r2 = float(1 - ((pr - tr) ** 2).sum() / ((tr - tr.mean()) ** 2).sum())
    plist = []
    for i in range(len(p)):
        q, r = p[i], t[i]
        if q.std() > 1e-8 and r.std() > 1e-8:
            plist.append(((q - q.mean()) * (r - r.mean())).mean() / (q.std() * r.std()).item())
    out = {"rmse": rmse, "mae": mae, "mae_celsius": mae_raw, "mape_percent": mape,
           "r2": r2, "pearson": float(np.mean(plist))}
    arrs = np.load(args.cache)
    doy = arrs["doys"][:len(p), -1] + 1
    months = month_of_doy(doy)
    e_std = (p - t).numpy()
    out["seasonal_rmse"] = {}
    for s in ("DJF", "MAM", "JJA", "SON"):
        idx = np.array([season_of_month(int(m)) == s for m in months])
        out["seasonal_rmse"][s] = float(np.sqrt((e_std[idx] ** 2).mean()))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
