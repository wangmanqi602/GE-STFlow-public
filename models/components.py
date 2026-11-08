"""Core building blocks of GE-STFlow.

AttnRes            : attention residual routing across depth
TargetGuidedAttention : target-guided spatial attention (query = target grid token)
TimestepEmbedder   : sinusoidal embedding of the flow time
EncoderLayer      : pre-norm Transformer layer with AttnRes routing
SeasonalEncoding  : sine/cosine day-of-year encoding
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
class AttnRes(nn.Module):
    """以可学习深度方向 softmax 注意力替代固定残差累积.

    h = Σ_l α_l · V_l,  α = softmax(q^T · RMSNorm(V) / √d_model)
    每个子层维护自己的查询向量 q, 在所有前序子层输出上加权聚合.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, d_model))
        self.rms_norm = nn.RMSNorm(d_model)
        self.scale = d_model ** 0.5

    def forward(self, prev_outputs: List[torch.Tensor], cur: torch.Tensor) -> torch.Tensor:
        # V: [K+1, B, T, d_model]
        V = torch.stack(prev_outputs + [cur], dim=0)
        attn = (self.q.view(1, 1, 1, -1) * self.rms_norm(V)).sum(-1) / self.scale
        alpha = F.softmax(attn, dim=0)                       # [K+1, B, T]
        return (alpha.unsqueeze(-1) * V).sum(dim=0)

class TargetGuidedAttention(nn.Module):
    """以目标位置为查询, 通过交叉注意力从邻域SST场中提取目标特定空间上下文.

    每个网格点 = 一个"token" (类比语言token): [SST值; 5维坐标特征] --MLP--> 嵌入.
    目标点为查询, 所有网格点为键/值. 输出 z^t ∈ R^{d_model}.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_coord: int = 5):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(1 + n_coord, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, sst: torch.Tensor, coords: torch.Tensor,
                tgt_idx: Tuple[int, int], land_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """sst: [B,H,W] (标准化), coords: [B,H,W,5] -> z: [B, d_model]

        land_mask: [B,H,W] bool, True=陆地/海冰 (与the reference setting约定一致, 直接作为
        key_padding_mask 屏蔽无效格点)."""
        B, H, W = sst.shape
        feat = torch.cat([sst.unsqueeze(-1), coords], dim=-1).view(B, H * W, -1)  # [B,P,6]
        tokens = self.token_proj(feat)                                            # [B,P,d]
        q = tokens[:, tgt_idx[0] * W + tgt_idx[1]: tgt_idx[0] * W + tgt_idx[1] + 1]  # 目标token
        kp = None if land_mask is None else land_mask.view(B, H * W)
        z, _ = self.cross_attn(q, tokens, tokens, key_padding_mask=kp)
        return z.squeeze(1)

class TimestepEmbedder(nn.Module):
    """扩散时间步 t∈[0,1] 的正弦嵌入 + MLP."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:      # t: [B]
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=-1)
        return self.mlp(emb)

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, int(d_model * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(d_model * mlp_ratio), d_model))
        self.attn_res = AttnRes(d_model)
        self.ffn_res = AttnRes(d_model)

    def forward(self, x: torch.Tensor, attn_history: List, ffn_history: List) -> torch.Tensor:
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = self.attn_res(attn_history, a)
        f = self.ffn(self.norm2(x))
        x = self.ffn_res(ffn_history, f)
        return x

class SeasonalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2, d_model)

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """doy: [B, T] (年积日) -> [B, T, d]"""
        r = 2 * math.pi * doy / 365.0
        return self.proj(torch.stack([torch.sin(r), torch.cos(r)], dim=-1))
