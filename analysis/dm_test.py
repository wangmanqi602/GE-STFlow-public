"""Diebold-Mariano test between two test exports (MSE loss difference).

Newey-West HAC variance with lag = T_out - 1 = 59, following the paper.
Positive DM statistic means the first model has lower loss.
"""
import argparse
import numpy as np, torch
from scipy import stats

def dm_test(d, lag=59):
    d = np.asarray(d, dtype=np.float64); n = len(d)
    db = d - d.mean()
    s = (db ** 2).sum() / n
    for k in range(1, lag + 1):
        s += 2 * (db[k:] * db[: n - k]).sum() / n * (1 - k / (lag + 1))
    stat = d.mean() / np.sqrt(max(s / n, 1e-12))
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))

def mse_per_sample(path):
    e = torch.load(path, map_location="cpu", weights_only=False)
    p = torch.as_tensor(e["pred"], dtype=torch.float64)
    t = torch.as_tensor(e["tgt"], dtype=torch.float64)
    return ((p - t) ** 2).mean(dim=1).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--lag", type=int, default=59)
    args = ap.parse_args()
    d = mse_per_sample(args.model_b) - mse_per_sample(args.model_a)
    stat, p = dm_test(d, args.lag)
    print(f"DM statistic = {stat:+.2f}, p = {p:.3e} (positive: model_a better)")

if __name__ == "__main__":
    main()
