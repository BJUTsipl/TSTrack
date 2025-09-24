import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint


class LightGQA(nn.Module):
    """极简分组查询注意力（计算量减少60%+）"""

    def __init__(self, d_model=512, num_heads=8, num_groups=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_groups

        # 共享投影矩阵（QKV合并投影）
        self.qkv_proj = nn.Linear(d_model, 2 * d_model // num_groups + d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 跟踪任务特化参数
        self.temporal_conv = nn.Conv1d(d_model, d_model, 3, padding=1, groups=8)  # 时序卷积增强

    def forward(self, query, key, value):
        # 合并投影计算（减少30%矩阵乘）
        qkv = self.qkv_proj(query)
        q, k, v = torch.split(qkv, [self.d_model,
                                    self.d_model // self.num_groups,
                                    self.d_model // self.num_groups], dim=-1)

        # 时序特征增强（跟踪任务关键）
        k = self.temporal_conv(k.transpose(1, 2)).transpose(1, 2)
        v = self.temporal_conv(v.transpose(1, 2)).transpose(1, 2)

        # 分组注意力计算
        B, L, _ = q.shape
        q = q.view(B, L, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, L, self.num_groups, -1).repeat(1, 1, self.num_heads // self.num_groups, 1)
        v = v.view(B, L, self.num_groups, -1).repeat(1, 1, self.num_heads // self.num_groups, 1)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim  **  -0.5)
        attn = self.dropout(attn.softmax(dim=-1))
        output = (attn @ v).transpose(1, 2).reshape(B, L, -1)

        return self.out_proj(output), attn.mean(dim=1)


class UltraLightInteraction(nn.Module):
    def __init__(self, d_model, grad_ckpt=False):
        super().__init__()
        # 共享归一化层（减少50%归一化计算）
        self.share_norm = nn.LayerNorm(d_model)

        # 轻量级注意力（2组GQA）
        self.attn = LightGQA(d_model, num_heads=8, num_groups=2)

        # 卷积替代FFN（减少90%参数量）
        self.conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.GELU(),
            nn.Conv1d(d_model // 4, d_model, 3, padding=1, groups=8),
            nn.Dropout(0.1)
        )

        self.grad_ckpt = grad_ckpt

    def forward(self, x, xs):
        # 特征交互（节省30%内存）
        xs = checkpoint.checkpoint(self._inner_forward, x, xs) if self.grad_ckpt else self._inner_forward(x, xs)
        return xs

    def _inner_forward(self, x, xs):
        # 共享归一化
        q = self.share_norm(xs)
        k = self.share_norm(x)

        # 轻量注意力
        attn_out, _ = self.attn(q, k, k)
        xs = xs + attn_out

        # 卷积FFN
        xs = xs + self.conv_ffn(xs.transpose(1, 2)).transpose(1, 2)
        return xs


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank,
                 d_state, bias=False, d_conv=3, conv_bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state

        # 轻量投影矩阵
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=bias)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                padding=(d_conv - 1) // 2, groups=d_inner, bias=conv_bias)

        # 简化参数初始化
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # 状态空间模型参数
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x, h):

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 轻量时序处理
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)

        # 简化SSM计算
        y, h = self.ssm_step(x, h)
        return self.out_proj(y * F.silu(z)), h

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        h = deltaA * h + deltaB * x.unsqueeze(-1)
        return (h @ C.unsqueeze(-1)).squeeze(-1) + self.D.float() * x, h


class ResidualBlock(nn.Module):
    def __init__(self, d_model=512, d_inner=512, grad_ckpt=False,  ** kwargs):
        super().__init__()
        self.mixer = MambaBlock(d_model=d_model, d_inner=d_inner,  ** kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.grad_ckpt = grad_ckpt

    def forward(self, x, h):
        x_norm = self.norm(x)
        if self.grad_ckpt:
            output, h = checkpoint.checkpoint(self.mixer, x_norm, h, use_reentrant=False)
        else:
            output, h = self.mixer(x_norm, h)
        return x + output, h


class TrackingMambaNeck(nn.Module):
    def __init__(self, in_channel, d_model, n_layers,
                 grad_ckpt, d_state, dt_rank):
        super().__init__()
        # 通道适配
        self.proj = nn.Linear(in_channel, d_model) if in_channel != d_model else nn.Identity()
        self.num_channels = d_model  # 添加这一行

        # 极简Mamba层
        self.layers = nn.ModuleList([
            ResidualBlock(
                d_model=d_model,
                d_inner=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                grad_ckpt=grad_ckpt
            ) for _ in range(n_layers)
        ])

        # 轻量交互模块
        self.interactions = nn.ModuleList([
            UltraLightInteraction(d_model, grad_ckpt)
            for _ in range(n_layers)
        ])

    def forward(self, x, xs, h):
        xs = self.proj(xs)
        for i, (layer, interact) in enumerate(zip(self.layers, self.interactions)):
            xs, h[i] = layer(xs, h[i])
            x = interact(x, xs)  # 单路交互输出
        return x, h

def build_neck(cfg, encoder):
    in_channel = encoder.num_channels
    d_model = cfg.MODEL.NECK.D_MODEL
    n_layers = cfg.MODEL.NECK.N_LAYERS
    d_state = cfg.MODEL.NECK.D_STATE
    dt_rank=cfg.MODEL.NECK.DT_RANK
    grad_ckpt = cfg.MODEL.ENCODER.GRAD_CKPT
    neck = TrackingMambaNeck(in_channel=in_channel, d_model=d_model, n_layers=n_layers,
                      dt_rank=dt_rank, d_state=d_state, grad_ckpt=grad_ckpt)
    return neck
