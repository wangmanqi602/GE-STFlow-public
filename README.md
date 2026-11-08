# GE-STFlow

Official code for **"GE-STFlow: Generalized-Embedding-Guided Spatiotemporal
Reconstruction and Flow Matching for Long-Range Sea Surface Temperature
Forecasting"** (Manqi Wang, Yin Wang, Yuxiang Zhou; submitted to IEEE
Geoscience and Remote Sensing Letters).

GE-STFlow reconstructs a target-location state from a 30-day SST field and
generates the next 60 days jointly with conditional residual flow matching:

- **Encoder** — target-guided spatial attention (query at the target grid
  point, keys/values from the neighboring field) followed by temporal
  Transformer layers with attention residual routing, producing a conditional
  memory `M`.
- **Decoder** — a value-space flow-matching decoder conditioned on `M`, the
  flow time, and the seasonal phase through a single conditioning vector with
  adaptive layer normalization (adaLN).
- **Residual anchor** — flow matching is performed on the residual
  `r = Y - y_anchor`, and the last observed value `y_anchor` is added back at
  the output.

## 1. Environment

- Ubuntu 24.04.3 LTS, Python 3.13.11
- PyTorch 2.13.0 (CUDA 13.0), NumPy 2.5.1, SciPy 1.18.0
- pandas 3.0.5, xarray 2026.7.0, netCDF4 1.7.4, Matplotlib 3.11.1
- 2 x NVIDIA GeForce RTX 5070 Ti (16 GB); driver 580.126.09

```bash
pip install -r requirements.txt
```

## 2. Data

Download the NOAA OISST v2.1 daily product (0.25 deg) from
https://www.ncei.noaa.gov/products/optimum-interpolation-sst and place the
yearly files as:

```
data/oisst/sst.day.mean.2021.nc
...
data/oisst/sst.day.mean.2025.nc
```

Build the cache:

```bash
export SST_DATA_DIR=./data/oisst
export SST_CACHE_DIR=./data_cache
python dataset.py
```

Default protocol: 15-25 N, 165-175 E; 5x5 degree windows with 0.5 degree
stride; T_in = 30 days, T_out = 60 days; 2022-2023 train, 2024 validation,
2025 test. For another region:

```bash
SST_REGION="30,40,150,160" SST_CACHE_DIR=./data_cache/region2 python dataset.py
```

## 3. Training

Main model (60 epochs, as in the paper):

```bash
python train.py --epochs 60 --use_residual --seed 42 --out checkpoints/gestflow
```

Anchor-free ablation (Table II):

```bash
python train.py --epochs 60 --seed 42 --out checkpoints/anchor_free
```

## 4. Evaluation

```bash
python evaluate.py --ckpt checkpoints/gestflow/best_model.pt                    --out checkpoints/gestflow --seed 42 --save_export
python analysis/metrics.py --export checkpoints/gestflow/test_export.pt
python analysis/dm_test.py --model_a checkpoints/gestflow/test_export.pt                            --model_b checkpoints/anchor_free/test_export.pt
```

## 5. Baselines

`baselines/` contains PatchTST, DLinear, Autoformer, iTransformer, TimeMixer,
STDMamba, MG-TSD, DiffusionTS, TSDiff, climatology and persistence. Each
script uses the same split and writes `test_export.pt`:

```bash
python baselines/patchtst.py --epochs 30 --seed 42 --out results/patchtst
```

## 6. Results

`results/` contains the tables reported in the paper: main comparison
(Table I), seasonal RMSE (Table II), regional anchoring ablation (Table III),
and the DM tests.

## 7. Citation

```bibtex
@article{wang2026gestflow,
  title   = {GE-STFlow: Generalized-Embedding-Guided Spatiotemporal
             Reconstruction and Flow Matching for Long-Range Sea Surface
             Temperature Forecasting},
  author  = {Wang, Manqi and Wang, Yin and Zhou, Yuxiang},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026}
}
```

## 8. Contact

Corresponding author: Yin Wang (shuwangyin@163.com).
