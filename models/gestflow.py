"""GE-STFlow: generalized-embedding-guided spatiotemporal reconstruction
with conditional residual flow matching.

Encoder : target-guided spatial attention + temporal Transformer + AttnRes
Decoder : value-space flow-matching decoder conditioned on the memory M and a
          single conditioning vector (flow time + seasonal phase) through adaLN.
Anchor  : the diffusion target is the residual r = Y - y_anchor; the anchor is
          added back at the output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .components import (
    AttnRes, TargetGuidedAttention, TimestepEmbedder,
    EncoderLayer, SeasonalEncoding,
)

# ============================================================================
# 值空间解码器: adaLN 条件引入 (PSTDT 式, 去观测时间戳)
# ============================================================================

class ValueSpaceDecoder(nn.Module):
    """adaLN 去噪网络: query=带噪值嵌入, key/value=记忆M(无时间戳), 条件向量 c.

    输入:  z_t [B, T_out] + t [B](扩散步) + memory [B, T_in, d] + doy_start [B]
    输出:  x_hat [B, T_out] (干净的SST轨迹点估计)
    条件:  c = CondMLP([t_emb(t); SE(doy_start)]), 每层每子层生成 (γ, β, α).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, T_out: int = 60,
                 n_time_tokens: int = 4):   # n_time_tokens 保留参数位, v2.1 不再使用
        super().__init__()
        self.T_out = T_out
        self.d_model = d_model

        # 值 -> 嵌入: 每个预报位标量映射到 d_model (仅顺序位置编码, 无日历戳)
        self.val_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, T_out, d_model) * 0.02)

        # 条件向量: 扩散步嵌入 + 预报期起始日季节相位
        self.time_embedder = TimestepEmbedder(d_model)
        self.season_embed = SeasonalEncoding(d_model)
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(2 * d_model, d_model))

        # 自调节投影 (通道拼接)
        self.self_cond_proj = nn.Linear(d_model * 2, d_model)

        # adaLN 交叉注意力层: 条件向量 c 逐层生成 γ/β/α
        self.layers = nn.ModuleList([
            AdaLNCrossAttnBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # 输出: 嵌入 -> 标量
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, memory: torch.Tensor,
                doy_start: torch.Tensor,
                x_pred_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z_t: [B, T_out]; t: [B]; memory: [B, T_in, d]; doy_start: [B] -> x_hat: [B, T_out]"""
        # 值嵌入 (仅位置信息)
        x = self.val_proj(z_t.unsqueeze(-1)) + self.pos_enc          # [B, T_out, d]

        # 自调节
        if x_pred_prev is not None:
            sc = self.val_proj(x_pred_prev.unsqueeze(-1)) + self.pos_enc
            x = self.self_cond_proj(torch.cat([x, sc], dim=-1))

        # 条件向量 c = MLP([扩散步嵌入; 预报期季节相位]) — PSTDT 式单一条件向量
        t_emb = self.time_embedder(t)                               # [B, d]
        se = self.season_embed(doy_start.unsqueeze(1)).squeeze(1)   # [B, d]
        c = self.cond_mlp(torch.cat([t_emb, se], dim=-1))           # [B, d]

        for layer in self.layers:
            x = layer(x, memory, c)
        return self.head(x).squeeze(-1)                             # [B, T_out]


class AdaLNCrossAttnBlock(nn.Module):
    """自注意力 + 交叉注意力 + FFN, 每子层由条件向量 c 生成 (γ, β, α) 的
    自适应层归一化 (PSTDT/DiT 式 adaLN-zero), 取代固定 RMSNorm."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)), nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        # 输出 [γ1, β1, α1, γ2, β2, α2, γ3, β3, α3], 每段 d_model 维
        self.adaLN = nn.Linear(d_model, 9 * d_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        with torch.no_grad():
            self.adaLN.bias[0 * d_model: 1 * d_model] = 1.0   # γ1 = 1
            self.adaLN.bias[3 * d_model: 4 * d_model] = 1.0   # γ2 = 1
            self.adaLN.bias[6 * d_model: 7 * d_model] = 1.0   # γ3 = 1
            # β, α 保持 0 (adaLN-zero 稳定初始化)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g = self.adaLN(c).unsqueeze(1)                  # [B, 1, 9d]
        g1, b1, a1, g2, b2, a2, g3, b3, a3 = g.chunk(9, dim=-1)
        # 自注意力 (q=k=v 均调制)
        h = g1 * self.norm1(x) + b1
        x = x + a1 * self.self_attn(h, h, h)[0]
        # 交叉注意力 (仅 query 调制, 键值为无时间戳记忆 M)
        h = g2 * self.norm2(x) + b2
        x = x + a2 * self.cross_attn(h, cond, cond)[0]
        # FFN
        h = g3 * self.norm3(x) + b3
        x = x + a3 * self.ffn(h)
        return x


# ============================================================================
# ============================================================================

class GESTFlow(nn.Module):
    """值空间扩散 SST 预测模型 (去观测时间戳版).

    输入侧 (保留the reference setting):  SST场 --the reference setting--> 上下文序列 --编码器--> 记忆M (无日历戳)
    扩散侧 (值空间):  z_t ∈ R^60 --adaLN交叉Attn(M)--> x_hat ∈ R^60
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_encoder_layers: int = 4, n_denoiser_layers: int = 4,
                 T_in: int = 30, T_out: int = 60,
                 n_time_tokens: int = 4,
                 noise_scale: float = 1.0, t_eps: float = 5e-2,
                 use_residual: bool = False):
        super().__init__()
        self.d_model = d_model
        self.T_in = T_in
        self.T_out = T_out
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.use_residual = use_residual

        self.spatial_attn = TargetGuidedAttention(d_model, n_heads)
        self.pos_enc = nn.Parameter(torch.randn(1, T_in, d_model) * 0.02)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads) for _ in range(n_encoder_layers)
        ])
        self.enc_out_res = AttnRes(d_model)

        # 值空间去噪器 (adaLN 条件引入)
        self.denoiser = ValueSpaceDecoder(
            d_model, n_heads, n_denoiser_layers, T_out, n_time_tokens,
        )

    # ---- 输入侧 ----
    def encode_memory(self, sst_fields, coords, tgt_idx, land_mask, doys):
        B, T_in, H, W = sst_fields.shape
        ti, tj = tgt_idx
        last_sst = sst_fields[:, -1, ti, tj] if self.use_residual else None
        kp = land_mask
        zs = [self.spatial_attn(sst_fields[:, t], coords, tgt_idx, kp) for t in range(T_in)]
        z = torch.stack(zs, dim=1)
        z = z + self.pos_enc                     # 仅顺序信息, 无日历时间戳
        hist = [z]
        for layer in self.encoder:
            z = layer(z, hist.copy(), hist.copy())
            hist.append(z)
        doy_start = doys[:, -1] + 1.0            # 预报期起始年积日 (解码器条件向量用)
        return self.enc_out_res(hist, z), last_sst, doy_start

    # ---- 扩散侧 ----
    def denoise(self, z_t: torch.Tensor, t: torch.Tensor, memory: torch.Tensor,
                doy_start: torch.Tensor,
                x_pred_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.denoiser(z_t, t, memory, doy_start, x_pred_prev)

    def velocity(self, x_pred, z_t, t):
        return (x_pred - z_t) / (1 - t).clamp(min=self.t_eps).view(-1, 1)

    def sample_timesteps_ode(self, n_steps, p_mean=0.0, p_std=0.8, seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(n_steps - 1, generator=g, device=device) * p_std + p_mean
        steps = torch.sigmoid(z)
        return torch.cat([torch.zeros(1, device=device),
                          torch.sort(steps).values,
                          torch.ones(1, device=device)])

    # ---- 推理 ----
    @torch.no_grad()
    def sample(self, sst_fields, coords, tgt_idx, land_mask, doys,
               num_steps=32, method="ode", gamma=0.0, p_mean=0.0, p_std=0.8,
               seed: int = 0):
        B = sst_fields.shape[0]
        memory, last_sst, doy_start = self.encode_memory(
            sst_fields, coords, tgt_idx, land_mask, doys)
        rng = torch.Generator(device=sst_fields.device).manual_seed(seed)
        z = torch.randn(B, self.T_out, generator=rng, device=sst_fields.device) * self.noise_scale
        ts = self.sample_timesteps_ode(num_steps, p_mean, p_std, seed=seed, device=sst_fields.device)
        x_pred_prev = None
        for i in range(num_steps):
            t_cur, t_nxt = ts[i].item(), ts[i + 1].item()
            tb = torch.full((B,), t_cur, device=sst_fields.device)
            x_pred = self.denoise(z, tb, memory, doy_start, x_pred_prev)
            v = self.velocity(x_pred, z, tb)
            z = z + (t_nxt - t_cur) * v
            x_pred_prev = x_pred.detach()
        pred = z
        if self.use_residual and last_sst is not None:
            pred = pred + last_sst.unsqueeze(-1)
        return pred
