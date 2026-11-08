"""Data pipeline for GE-STFlow.

Reads yearly NOAA OISST v2.1 NetCDF files, extracts 5x5 degree sub-windows,
builds 30-day input / 60-day target samples, and stores a z-scored cache.

Environment variables:
    SST_DATA_DIR   directory containing sst.day.mean.YYYY.nc
    SST_CACHE_DIR  cache directory (default ./data_cache)
    SST_REGION     "lat_lo,lat_hi,lon_lo,lon_hi" (default 15,25,165,175)
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SST_CODE_DIR = Path(__file__).resolve().parent / "data_utils"
if str(SST_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SST_CODE_DIR))

from config import CONFIG  # noqa: E402
from data_extract import (  # noqa: E402
    open_sst_dataset, split_by_time, compute_mean_std,
    make_start_indices, generate_sub_windows,
)
from spatial_attention import build_coord_features  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _apply_env_overrides():
    """环境变量覆盖 (用于换海域泛化实验, 默认行为不变):
    SST_REGION = "lat_lo,lat_hi,lon_lo,lon_hi"   如 "30,40,150,160"
    SST_CACHE_DIR = 缓存目录 (默认 ./data_cache)"""
    global CACHE_DIR
    if os.environ.get("SST_CACHE_DIR"):
        CACHE_DIR = Path(os.environ["SST_CACHE_DIR"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SST_REGION"):
        la_lo, la_hi, lo_lo, lo_hi = [float(x) for x in os.environ["SST_REGION"].split(",")]
        CONFIG.data.lat_value_range = [la_lo, la_hi]
        CONFIG.data.lon_value_range = [lo_lo, lo_hi]


_apply_env_overrides()


def build_datasets(force_rebuild: bool = False):
    """加载原始 .nc → 切窗 → 标准化 → (缓存) → 返回 dict of np arrays + meta."""
    d = CONFIG.data
    cache_files = {
        split: CACHE_DIR / f"samples_{split}.npz" for split in ("train", "val", "test")
    }
    meta_path = CACHE_DIR / "samples_meta.npz"
    if not force_rebuild and all(f.exists() for f in cache_files.values()) and meta_path.exists():
        print("加载缓存数据:", ", ".join(str(f) for f in cache_files.values()))
        arrs = {split: np.load(f) for split, f in cache_files.items()}
        m = np.load(meta_path)
        coord = m["coord"]
        arrs["coord"] = coord
        meta = {"mean": float(m["mean"]), "std": float(m["std"]),
                "n_windows": int(m["n_windows"]),
                "target_lat": float(m["target_lat"][0]),
                "target_lon": float(m["target_lon"][0])}
        arrs["meta"] = meta
        return arrs, meta["mean"], meta["std"]

    print("加载原始 SST 数据 ...")
    ds, sst = open_sst_dataset(
        d.data_path, variable_name=d.variable_name,
        lat_range=d.lat_value_range, lon_range=d.lon_value_range,
    )
    train_sst, val_sst, test_sst = split_by_time(
        sst, train_time=d.train_time, val_time=d.val_time, test_time=d.test_time,
    )
    mean, std = compute_mean_std(train_sst)
    print(f"  区域: lat {d.lat_value_range}, lon {d.lon_value_range}, mean={mean:.4f}, std={std:.4f}")

    lat_vals = sst["lat"].values.astype("float32")
    lon_vals = sst["lon"].values.astype("float32")
    windows = generate_sub_windows(
        lat_values=lat_vals, lon_values=lon_vals,
        lat_span=d.sub_lat_span, lon_span=d.sub_lon_span,
        stride_lat=d.stride_lat, stride_lon=d.stride_lon,
    )
    print(f"  子窗口数量: {len(windows)}")

    # 坐标特征模板: 所有子窗口同形状(20×20), 目标点同为窗口中心 → 坐标特征一致
    sub_h = windows[0]["lat_end"] - windows[0]["lat_start"]
    sub_w = windows[0]["lon_end"] - windows[0]["lon_start"]
    sub_lat = lat_vals[windows[0]["lat_start"]:windows[0]["lat_end"]]
    sub_lon = lon_vals[windows[0]["lon_start"]:windows[0]["lon_end"]]
    coord = build_coord_features(
        sub_lat, sub_lon,
        target_lat_idx=sub_h // 2, target_lon_idx=sub_w // 2,
    )  # [5, H, W]
    coord = coord.permute(1, 2, 0).numpy()  # [H, W, 5]

    # 目标点经纬度 (每个窗口中心)
    target_lat = np.array([float(sub_lat[sub_h // 2])], dtype=np.float32)
    target_lon = np.array([float(sub_lon[sub_w // 2])], dtype=np.float32)

    def precompute_split(split_sst, starts):
        """遍历所有 (start, window), 提取 [src场, 目标轨迹, 陆地mask]."""
        src_all, tgt_all, land_all = [], [], []
        # 大区域整体取一次 (标准化), 再按窗口切片
        raw_all = split_sst.values  # [T, H, W] float (可能含NaN)
        norm_all = (raw_all - mean) / std
        for start in starts:
            # 一次性切整个时间窗
            win = norm_all[start: start + d.t_in + d.t_out]  # [T_in+T_out, H, W]
            s = win[:d.t_in]
            t = win[d.t_in: d.t_in + d.t_out]
            # 逐窗口取中心点轨迹 + 子窗口场
            for w in windows:
                i0, i1 = w["lat_start"], w["lat_end"]
                j0, j1 = w["lon_start"], w["lon_end"]
                i_c = i0 + w["target_lat_idx"]
                j_c = j0 + w["target_lon_idx"]
                src_all.append(s[:, i0:i1, j0:j1])
                tgt_all.append(t[:, i_c, j_c])
                land_all.append(np.isnan(s[:, i0:i1, j0:j1]).any(axis=0))
            del win
        src = np.ascontiguousarray(np.stack(src_all))  # [N, T_in, H, W]
        tgt = np.ascontiguousarray(np.stack(tgt_all))  # [N, T_out]
        land = np.ascontiguousarray(np.stack(land_all))  # [N, H, W] bool
        return src, tgt, land

    # doy: 每个样本输入窗口逐天年积日 (用 pandas 时间转换)
    import pandas as pd

    def _make_doys(time_vals, starts):
        doys = np.zeros((len(starts) * len(windows), d.t_in), dtype=np.float32)
        for si, start in enumerate(starts):
            days = pd.to_datetime(time_vals[start:start + d.t_in])
            doys[si * len(windows): (si + 1) * len(windows)] = days.dayofyear.values
        return doys

    arrs = {}
    for split, split_sst in (("train", train_sst), ("val", val_sst), ("test", test_sst)):
        starts = make_start_indices(split_sst, d.t_in, d.t_out)
        print(f"  预计算 {split}: {len(starts)} 个起始日 × {len(windows)} 窗口 ...")
        src, tgt, land = precompute_split(split_sst, starts)
        doys = _make_doys(split_sst.time.values, starts)
        src = np.nan_to_num(src, nan=d.fill_value)
        tgt = np.nan_to_num(tgt, nan=d.fill_value)
        np.savez(cache_files[split], src=src, tgt=tgt, land=land, doys=doys)
        arrs[split] = np.load(cache_files[split])
        print(f"    {split}: src={src.shape}, tgt={tgt.shape}, land={land.shape}, doys={doys.shape}")

    # 公共元数据缓存 (坐标特征模板 + 统计量)
    meta_path = CACHE_DIR / "samples_meta.npz"
    np.savez(meta_path, coord=coord, mean=np.float32(mean), std=np.float32(std),
             n_windows=np.int32(len(windows)),
             target_lat=target_lat, target_lon=target_lon)
    arrs["coord"] = coord
    arrs["meta"] = {
        "mean": mean, "std": std, "n_windows": len(windows),
        "target_lat": target_lat[0], "target_lon": target_lon[0],
    }
    return arrs, mean, std


class STISSTDataset(Dataset):
    """PyTorch Dataset, 返回 GE-STFlow 训练/评估所需的样本."""

    def __init__(self, split: str, arrs: dict, t_in: int, t_out: int):
        self.split = split
        self.src = torch.from_numpy(arrs[split]["src"])      # [N, T_in, H, W]
        self.tgt = torch.from_numpy(arrs[split]["tgt"])      # [N, T_out]
        self.land = torch.from_numpy(arrs[split]["land"])    # [N, H, W] bool
        self.doys = torch.from_numpy(arrs[split]["doys"])    # [N, T_in]
        self.coord = torch.from_numpy(arrs["coord"])         # [H, W, 5]
        self.t_in = t_in
        self.t_out = t_out
        self.n_windows = int(arrs["meta"]["n_windows"])
        self.target_lat = float(arrs["meta"]["target_lat"])
        self.target_lon = float(arrs["meta"]["target_lon"])
        # 目标点 = 窗口中心 (所有子窗口同形状)
        self.target_idx = (self.src.shape[1] // 2, self.src.shape[2] // 2)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "sst_fields": self.src[idx],                 # [T_in, H, W]
            "coords": self.coord,                        # [H, W, 5]
            "land_mask": self.land[idx],                 # [H, W]
            "doys": self.doys[idx],                      # [T_in]
            "target_trajectory": self.tgt[idx],          # [T_out]
            "target_idx": self.target_idx,               # (H//2, W//2)
            "target_lat": self.target_lat,
            "target_lon": self.target_lon,
            "win_idx": idx % self.n_windows,             # 窗口索引 (评估/绘图用)
        }


def make_loaders(arrs: dict, batch_size: int, num_workers: int = 4,
                 drop_last_test: bool = True):
    """构造 train/val/test DataLoader (test 用 drop_last=True, 与 the reference setting 一致)."""
    from torch.utils.data import DataLoader

    d = CONFIG.data
    train_ds = STISSTDataset("train", arrs, d.t_in, d.t_out)
    val_ds = STISSTDataset("val", arrs, d.t_in, d.t_out)
    test_ds = STISSTDataset("test", arrs, d.t_in, d.t_out)

    def collate(batch):
        out = {}
        for k in batch[0]:
            v = [b[k] for b in batch]
            if k == "coords":
                out[k] = v[0].unsqueeze(0).repeat(len(batch), 1, 1, 1)
            elif k == "target_idx":
                out[k] = v[0]                    # 常量 tuple, 所有样本相同
            elif isinstance(v[0], (float, int)):
                out[k] = torch.tensor(v)
            else:
                out[k] = torch.stack(v)
        return out

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate,
                              drop_last=True, persistent_workers=True,
                              prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate,
                            drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate,
                             drop_last=drop_last_test)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import time
    t0 = time.time()
    arrs, mean, std = build_datasets()
    print(f"构建完成, 用时 {time.time()-t0:.1f}s, mean={mean:.4f}, std={std:.4f}")
    meta = arrs["meta"]
    print(f"meta: {meta}")
    for split in ("train", "val", "test"):
        a = arrs[split]
        print(f"  {split}: src {a['src'].shape}, tgt {a['tgt'].shape}, "
              f"land {a['land'].shape}, doys {a['doys'].shape}")
