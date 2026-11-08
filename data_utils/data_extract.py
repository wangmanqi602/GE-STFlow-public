import glob
from typing import Optional, Tuple
import numpy as np
import torch
import xarray as xr


def open_sst_dataset(
    data_path: str,
    variable_name: str = "sst",
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
):

    if data_path.endswith(".nc"):
        ds = xr.open_dataset(data_path)
        if lat_range is not None:
            ds = ds.sel(lat=slice(lat_range[0], lat_range[1]))
        if lon_range is not None:
            ds = ds.sel(lon=slice(lon_range[0], lon_range[1]))
    else:
        file_list = sorted(glob.glob(f"{data_path}/*.nc"))

        if len(file_list) == 0:
            raise FileNotFoundError(f"在 {data_path} 中没有找到 .nc 文件。")

        print(f"找到 {len(file_list)} 个 .nc 文件：")
        for file in file_list:
            print(file)

        # 逐个文件加载到内存，存取 numpy，绕开 dask 多文件并发读取的 HDF 错误
        # 在加载阶段就裁剪区域，避免读全分辨率数据
        arrays = []
        times = []
        lats = None
        lons = None
        for f in file_list:
            try:
                with xr.open_dataset(f) as ds_part:
                    sst_part = ds_part[variable_name]
                    if lat_range is not None:
                        sst_part = sst_part.sel(lat=slice(lat_range[0], lat_range[1]))
                    if lon_range is not None:
                        sst_part = sst_part.sel(lon=slice(lon_range[0], lon_range[1]))
                    arrays.append(np.asarray(sst_part.values))
                    times.append(np.asarray(sst_part.time.values))
                    if lats is None:
                        lats = np.asarray(sst_part.lat.values)
                        lons = np.asarray(sst_part.lon.values)
            except (RuntimeError, OSError) as e:
                print(f"  跳过损坏文件: {f} ({e})")

        if len(arrays) == 0:
            raise RuntimeError("没有可用的 .nc 文件。")

        all_data = np.concatenate(arrays, axis=0)
        all_times = np.concatenate(times, axis=0)

        ds = xr.Dataset(
            {variable_name: (["time", "lat", "lon"], all_data)},
            coords={"time": all_times, "lat": lats, "lon": lons},
        )

    if variable_name not in ds.data_vars:
        raise KeyError(
            f"没有找到变量 {variable_name}。"
            f"当前文件中的数据变量有：{list(ds.data_vars)}"
        )

    sst = ds[variable_name]

    if sst.dims != ("time", "lat", "lon"):
        raise ValueError(
            f"{variable_name} 的维度顺序是 {sst.dims}，"
        )   

    return ds, sst


def select_region(
    sst,
    lat_index_range: Optional[Tuple[int, int]] = None,
    lon_index_range: Optional[Tuple[int, int]] = None,
    lat_value_range: Optional[Tuple[float, float]] = None,
    lon_value_range: Optional[Tuple[float, float]] = None,
):
    """
    选择 SST 的空间区域。
    支持两种方式：
    方式一：按索引选择
        lat_index_range=(300, 400)
        lon_index_range=(500, 600)
    方式二：按真实经纬度选择
        lat_value_range=(-20, 20)
        lon_value_range=(120, 180)
    """

    if lat_index_range is not None and lon_index_range is not None:
        lat_start, lat_end = lat_index_range
        lon_start, lon_end = lon_index_range

        region_sst = sst.isel(
            lat=slice(lat_start, lat_end),
            lon=slice(lon_start, lon_end)
        )

        return region_sst

    if lat_value_range is not None and lon_value_range is not None:
        lat_min, lat_max = lat_value_range
        lon_min, lon_max = lon_value_range

        region_sst = sst.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )

        return region_sst

    return sst


def split_by_time(
    sst,
    train_time: Tuple[str, str],
    val_time: Tuple[str, str],
    test_time: Tuple[str, str],
):
    train_start, train_end = data_transfer(sst, train_time)
    val_start, val_end = data_transfer(sst, val_time)
    test_start, test_end = data_transfer(sst, test_time)

    train_sst = sst.isel(time=slice(train_start, train_end))
    val_sst = sst.isel(time=slice(val_start, val_end))
    test_sst = sst.isel(time=slice(test_start, test_end))

    return train_sst, val_sst, test_sst


def data_transfer(
    sst,
    date_range: Tuple[str, str],
) -> Tuple[int, int]:
    """
    把日期字符串区间转换为 time 维度的索引区间。

    例如:
        ("2010-01-01", "2018-12-31") -> (start_idx, end_idx_exclusive)
    """
    if "time" not in sst.dims:
        raise KeyError("当前 sst 不包含 time 维度。")

    start_date, end_date = date_range
    if start_date > end_date:
        raise ValueError(f"开始日期晚于结束日期: {date_range}")

    time_values = np.asarray(sst["time"].values)
    if time_values.size == 0:
        raise ValueError("time 维度为空，无法进行日期到索引的转换。")

    start_dt = np.datetime64(start_date)
    end_dt = np.datetime64(end_date)

    # 左闭右闭的日期区间 -> 左闭右开的索引区间
    start_idx = int(np.searchsorted(time_values, start_dt, side="left"))
    end_idx_exclusive = int(np.searchsorted(time_values, end_dt, side="right"))

    if start_idx >= len(time_values) or end_idx_exclusive <= 0:
        raise ValueError(
            f"日期区间 {date_range} 超出数据范围: "
            f"{str(time_values[0])} ~ {str(time_values[-1])}"
        )

    start_idx = max(0, start_idx)
    end_idx_exclusive = min(len(time_values), end_idx_exclusive)

    if start_idx >= end_idx_exclusive:
        raise ValueError(f"日期区间 {date_range} 转换后为空，请检查参数。")

    return start_idx, end_idx_exclusive


def generate_sub_windows(
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    lat_span: float,
    lon_span: float,
    stride_lat: float,
    stride_lon: float,
):
    """
    在大区域内滑动生成子窗口列表。

    每个子窗口大小由 lat_span / lon_span（度）和网格分辨率决定。
    目标点默认取子窗口中心。

    返回:
        list[dict], 每个 dict 含:
            lat_start, lat_end, lon_start, lon_end: 子窗口在大区域内的索引范围
            target_lat_idx, target_lon_idx:          目标点在子窗口内的行列索引
    """
    lat_res = float(lat_values[1] - lat_values[0])
    lon_res = float(lon_values[1] - lon_values[0])

    sub_h = int(round(lat_span / lat_res))
    sub_w = int(round(lon_span / lon_res))
    step_h = max(1, int(round(stride_lat / lat_res)))
    step_w = max(1, int(round(stride_lon / lon_res)))

    H = len(lat_values)
    W = len(lon_values)

    target_lat_idx = sub_h // 2
    target_lon_idx = sub_w // 2

    windows = []
    for i in range(0, H - sub_h + 1, step_h):
        for j in range(0, W - sub_w + 1, step_w):
            windows.append({
                "lat_start": i,
                "lat_end": i + sub_h,
                "lon_start": j,
                "lon_end": j + sub_w,
                "target_lat_idx": target_lat_idx,
                "target_lon_idx": target_lon_idx,
            })

    return windows


def compute_mean_std(sst):
    """
    计算训练集 SST 的均值和标准差。
    """

    values = sst.values

    mean = np.nanmean(values)
    std = np.nanstd(values)

    if np.isnan(mean):
        raise ValueError("mean 是 NaN，说明当前区域可能全部是缺失值。")

    if np.isnan(std) or std == 0:
        raise ValueError("std 是 NaN 或 0，说明当前区域数据异常，无法标准化。")

    return float(mean), float(std)


def get_num_samples(sst, t_in: int, t_out: int):
    """
    计算当前时间段可以构造多少个滑窗样本。

    total_time = 总时间长度

    一个样本需要：
        t_in 天输入 + t_out 天输出

    样本数：
        total_time - t_in - t_out + 1
    """

    total_time = sst.sizes["time"]
    num_samples = total_time - t_in - t_out + 1

    if num_samples <= 0:
        raise ValueError(
            f"当前时间长度为 {total_time}，"
            f"但 t_in={t_in}, t_out={t_out}，无法构造样本。"
        )

    return num_samples


def make_start_indices(sst, t_in: int, t_out: int):
    """
    生成所有合法的 start 索引。

    每一个 start 对应一个样本：

    src = [start, start + t_in)
    tgt = [start + t_in, start + t_in + t_out)
    """

    num_samples = get_num_samples(sst, t_in, t_out)
    return list(range(num_samples))


def make_one_sample(
    sst,
    start: int,
    t_in: int,
    t_out: int,
    mean: float,
    std: float,
    target_lat_idx: Optional[int] = None,
    target_lon_idx: Optional[int] = None,
    fill_value: float = 0.0,
):
    """
    构造一个 SST 预测样本。

    输入：
    sst   : xarray.DataArray，维度为 [time, lat, lon]
    start : 样本起始时间索引
    t_in  : 输入序列长度
    t_out : 输出序列长度
    mean  : 训练集均值
    std   : 训练集标准差

    返回：
    src          : [T_in, 1, H, W]
    tgt_in       : [T_out, 1]
    tgt_out      : [T_out, 1]
    src_time_idx : [T_in]
    tgt_time_idx : [T_out]
    src_season   : [T_in, 2]
    tgt_season   : [T_out, 2]
    land_mask    : [H, W]  bool, True = 陆地/海冰
    """

    src_raw = sst.isel(
        time=slice(start, start + t_in)
    ).values   # [T_in, H, W], NaN = 陆地/海冰

    tgt_raw = sst.isel(
        time=slice(start + t_in, start + t_in + t_out)
    ).values

    # 陆地/海冰 mask: True = 无效像素，应在空间注意力中屏蔽
    land_mask = torch.from_numpy(
        np.isnan(src_raw).any(axis=0)  # [H, W], 任一天 NaN 即陆地
    )

    # 标准化
    src = (src_raw - mean) / std
    tgt = (tgt_raw - mean) / std

    # 陆地/海冰填 0（已通过 land_mask 标记，不会参与注意力计算）
    src = np.nan_to_num(src, nan=fill_value)
    tgt = np.nan_to_num(tgt, nan=fill_value)

    # 转成 PyTorch Tensor
    # 原始形状: [T, H, W]
    # 加 channel 维后: [T, 1, H, W]
    src = torch.tensor(src, dtype=torch.float32).unsqueeze(1)
    tgt = torch.tensor(tgt, dtype=torch.float32).unsqueeze(1)

    # 默认目标点取当前区域中心点
    # src/tgt 形状: [T, 1, H, W]
    _, _, h, w = src.shape
    if target_lat_idx is None:
        target_lat_idx = h // 2
    if target_lon_idx is None:
        target_lon_idx = w // 2

    if not (0 <= target_lat_idx < h and 0 <= target_lon_idx < w):
        raise IndexError(
            f"目标点索引超出区域范围: target=({target_lat_idx}, {target_lon_idx}),"
            f" region_shape=({h}, {w})"
        )

    # 抽取目标点的一维时间序列（去掉 channel 维）
    src_point = src[:, 0, target_lat_idx, target_lon_idx]  # [T_in]
    tgt_point = tgt[:, 0, target_lat_idx, target_lon_idx]  # [T_out]

    # decoder 输入: [最后一个历史值, 未来真值前 T_out-1 个]
    tgt_in = torch.cat([src_point[-1:], tgt_point[:-1]], dim=0)         # [T_out]
    tgt_out = tgt_point                                                   # [T_out]

    tgt_in = tgt_in.float().unsqueeze(-1)    # [T_out, 1]
    tgt_out = tgt_out.float().unsqueeze(-1)  # [T_out, 1]

    src_time_idx = torch.arange(t_in, dtype=torch.long)
    tgt_time_idx = torch.arange(t_out, dtype=torch.long)

    # 季节编码：sin/cos(day_of_year)
    doy_src = sst.time.isel(time=slice(start, start + t_in)).dt.dayofyear.values.astype(np.float32)
    doy_tgt = sst.time.isel(time=slice(start + t_in, start + t_in + t_out)).dt.dayofyear.values.astype(np.float32)
    angle_src = 2.0 * np.pi * doy_src / 365.0
    angle_tgt = 2.0 * np.pi * doy_tgt / 365.0
    src_season = torch.tensor(np.stack([np.sin(angle_src), np.cos(angle_src)], axis=-1), dtype=torch.float32)
    tgt_season = torch.tensor(np.stack([np.sin(angle_tgt), np.cos(angle_tgt)], axis=-1), dtype=torch.float32)

    return src, tgt_in, tgt_out, src_time_idx, tgt_time_idx, src_season, tgt_season, land_mask

