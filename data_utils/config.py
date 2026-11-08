"""
训练与数据相关参数集中在这里修改即可。

说明:
- 时间划分使用日期字符串 (与 data_extract.data_transfer 一致)。
- 空间区域用 lat_value_range + lon_value_range 定义大范围采集区域，内部滑窗切分子窗口。
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DataConfig:
    # 单个 .nc 文件路径，或包含多个 .nc 的目录
    data_path: str = os.environ.get("SST_DATA_DIR", "./data/oisst")
    variable_name: str = "sst"

    train_time: Tuple[str, str] = ('2022-01-01', '2023-12-31')
    val_time: Tuple[str, str] = ('2024-01-01', '2024-12-31')
    test_time: Tuple[str, str] = ('2025-01-01', '2025-12-31')

    # 大数据采集区域（经纬度），滑窗在此范围内采样子区域
    lat_value_range: Tuple[float, float] = (15.0, 25.0)
    lon_value_range: Tuple[float, float] = (165.0, 175.0)

    # 滑窗子区域大小（度）
    sub_lat_span: float = 5.0
    sub_lon_span: float = 5.0
    # 滑窗步长（度）
    stride_lat: float = 0.5
    stride_lon: float = 0.5

    t_in: int = 30
    t_out: int = 60

    batch_size: int = 64
    num_workers: int = 2
    fill_value: float = 0.0


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "relu"
    normalize_before: bool = False
    max_len: int = 512
    # 空间注意力 (V1)
    use_spatial_attention: bool = True
    spatial_nhead: int = 4
    # Attention Residuals
    use_attn_res: bool = True
    # 空间上下文：True=SpatialAttention/全 patch；False=仅中心点
    use_spatial_context: bool = True
    # 季节编码
    use_seasonal_encoding: bool = True


@dataclass
class TrainConfig:
    lr: float = 1e-3
    num_epochs: int = 20
    grad_clip: Optional[float] = 1.0
    save_dir: str = "./checkpoints/baseline"
    use_ar_val: bool = True  # True=自回归验证, False=teacher forcing 验证


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


CONFIG = Config(
    data=DataConfig(),
    model=ModelConfig(),
    train=TrainConfig(),
)
