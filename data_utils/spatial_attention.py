from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor


def build_coord_features(
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    target_lat_idx: int,
    target_lon_idx: int,
) -> Tensor:
    """
    构建 5 通道空间坐标特征。

    lat_values: [H]  裁剪区域内的真实纬度值
    lon_values: [W]  裁剪区域内的真实经度值
    target_lat_idx, target_lon_idx: 目标点在裁剪区域内的行列索引

    返回: [5, H, W]
        ch0: lat_norm   当前格点纬度归一化到 [-1, 1]
        ch1: lon_norm   当前格点经度归一化到 [-1, 1]
        ch2: delta_lat  相对目标点的纬度差 (归一化坐标)
        ch3: delta_lon  相对目标点的经度差 (归一化坐标)
        ch4: distance   到目标点的欧氏距离 (归一化坐标空间)
    """
    H = len(lat_values)
    W = len(lon_values)

    lat_min, lat_max = lat_values.min(), lat_values.max()
    lon_min, lon_max = lon_values.min(), lon_values.max()
    # 单格点裁剪时跨度为 0，避免除零
    lat_span = float(lat_max - lat_min) or 1.0
    lon_span = float(lon_max - lon_min) or 1.0

    lat_grid = np.tile(lat_values[:, None], (1, W))  # [H, W]
    lon_grid = np.tile(lon_values[None, :], (H, 1))  # [H, W]

    lat_norm = 2.0 * (lat_grid - lat_min) / lat_span - 1.0  # [H, W]
    lon_norm = 2.0 * (lon_grid - lon_min) / lon_span - 1.0  # [H, W]

    target_lat_norm = lat_norm[target_lat_idx, target_lon_idx]
    target_lon_norm = lon_norm[target_lat_idx, target_lon_idx]

    delta_lat = lat_norm - target_lat_norm
    delta_lon = lon_norm - target_lon_norm
    distance = np.sqrt(delta_lat ** 2 + delta_lon ** 2)

    coord = np.stack([lat_norm, lon_norm, delta_lat, delta_lon, distance], axis=0)
    return torch.tensor(coord, dtype=torch.float32)


class TargetGuidedSpatialAttention(nn.Module):
    """
    目标点导向的空间多头注意力。

    在「每天 Q=1、K/V=P」的 V1 流程上，每个空间点用 6 维特征再投影到 d_model：
        [SST, lat_norm, lon_norm, delta_lat, delta_lon, distance]
    其中后 5 维由 build_coord_features 从真实经纬度生成（相对文字版「仅 SST 标量」的扩展）。

    输入:
        src_sst:        [B, T_in, 1, H, W]
        coord_features: [5, H, W]    (由 build_coord_features 构建)

    输出:
        z:             [B, T_in, d_model]
        attn_weights:  [B, T_in, spatial_nhead, H, W]  (仅 need_weights=True)
    """

    def __init__(
        self,
        h: int,
        w: int,
        d_model: int,
        spatial_nhead: int = 4,
        dropout: float = 0.1,
        coord_features: Optional[Tensor] = None,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.num_points = h * w
        self.d_model = d_model

        if d_model % spatial_nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) 必须能被 spatial_nhead ({spatial_nhead}) 整除。"
            )

        # 6 通道 → d_model 的小型 MLP
        self.value_proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=spatial_nhead,
            dropout=dropout,
            batch_first=True,
        )

        if coord_features is not None:
            self.register_buffer("coord_features", coord_features)
        else:
            self.coord_features = None

    def set_coord_features(self, coord_features: Tensor):
        """后续设置坐标特征（会注册为 buffer，自动跟随 model.to(device)）。"""
        self.register_buffer("coord_features", coord_features)

    def _check_shapes(self, src_sst: Tensor, coord: Tensor,
                      target_lat_idx: int, target_lon_idx: int):
        B, T_in, C, H, W = src_sst.shape
        if C != 1:
            raise ValueError(f"src_sst 通道数应为 1，实际 {C}")
        if H != self.h or W != self.w:
            raise ValueError(f"期望空间尺寸 ({self.h},{self.w})，实际 ({H},{W})")
        if coord.shape != (5, H, W):
            raise ValueError(f"coord_features 形状应为 (5,{H},{W})，实际 {coord.shape}")
        if not (0 <= target_lat_idx < H and 0 <= target_lon_idx < W):
            raise IndexError(f"目标点索引越界: ({target_lat_idx},{target_lon_idx}), 区域 ({H},{W})")

    def forward(
        self,
        src_sst: Tensor,
        target_lat_idx: int,
        target_lon_idx: int,
        need_weights: bool = False,
        land_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, T_in, _, H, W = src_sst.shape

        if self.coord_features is None:
            raise RuntimeError("coord_features 未设置，请在构造时传入或调用 set_coord_features()")
        coord = self.coord_features  # [5, H, W]

        self._check_shapes(src_sst, coord, target_lat_idx, target_lon_idx)

        # 1. 扩展坐标特征: [5, H, W] → [B, T_in, 5, H, W]
        coord = coord.unsqueeze(0).unsqueeze(0).expand(B, T_in, -1, -1, -1)

        # 2. 拼接 SST + 坐标: [B, T_in, 1+5=6, H, W]
        x = torch.cat([src_sst, coord], dim=2)

        # 3. 空间点展平: [B, T_in, 6, H, W] → [B*T_in, P, 6]
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, T_in, H, W, 6]
        x = x.view(B * T_in, self.num_points, 6)

        # 4. 6 通道 → d_model
        h = self.value_proj(x)  # [B*T_in, P, d_model]

        # 5. 目标点作为 Query
        target_idx = target_lat_idx * W + target_lon_idx
        q = h[:, target_idx:target_idx + 1, :]  # [B*T_in, 1, d_model]
        k = v = h                                # [B*T_in, P, d_model]

        # 5.5 构建 key_padding_mask：屏蔽陆地/海冰像素
        key_padding_mask = None
        if land_mask is not None:
            # land_mask: [B, H, W] (来自 DataLoader collate 堆叠)
            # 扩展为 [B*T_in, P]
            key_padding_mask = (
                land_mask.unsqueeze(1)            # [B, 1, H, W]
                .expand(-1, T_in, -1, -1)         # [B, T_in, H, W]
                .reshape(B * T_in, self.num_points)  # [B*T_in, P]
            )

        # 6. 空间多头注意力
        attn_output, attn_weights = self.spatial_attn(
            q, k, v,
            need_weights=need_weights,
            average_attn_weights=False if need_weights else True,
            key_padding_mask=key_padding_mask,
        )
        # attn_output: [B*T_in, 1, d_model]

        # 7. 还原时间序列格式: [B, T_in, d_model]
        z = attn_output.reshape(B, T_in, self.d_model)

        # 8. 注意力权重 reshape
        if need_weights and attn_weights is not None:
            # average_attn_weights=False: [B*T_in, spatial_nhead, 1, P]
            attn_weights = attn_weights.squeeze(2).reshape(B, T_in, -1, H, W)
            # [B, T_in, spatial_nhead, H, W]
            return z, attn_weights

        return z, None
