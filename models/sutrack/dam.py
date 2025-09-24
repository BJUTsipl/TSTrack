import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """使用深度可分离卷积的空间注意力模块"""

    def __init__(self):
        super().__init__()
        # 深度可分离卷积结构
        self.depthwise = nn.Conv2d(2, 2, kernel_size=5,
                                   padding=2, groups=2, bias=False)  # 深度卷积
        self.pointwise = nn.Conv2d(2, 1, kernel_size=1, bias=False)  # 点卷积
        self.sigmoid = nn.Sigmoid()

        # 初始化参数（保持GAP/GMP特性）
        nn.init.dirac_(self.depthwise.weight[0:1])  # 初始化GMP路径
        nn.init.dirac_(self.depthwise.weight[1:2])  # 初始化GAP路径
        nn.init.constant_(self.pointwise.weight, 0.5)  # 平衡权重

    def forward(self, x):
        # 输入形状: [B,3,H,W]
        gmp = x.amax(dim=1, keepdim=True)  # 全局最大池化 [B,1,H,W]
        gap = x.mean(dim=1, keepdim=True)  # 全局平均池化 [B,1,H,W]
        concat = torch.cat([gmp, gap], dim=1)  # 拼接 [B,2,H,W]

        # 深度可分离卷积
        x = self.depthwise(concat)  # [B,2,H,W]
        x = self.pointwise(x)  # [B,1,H,W]
        return self.sigmoid(x)  # [B,1,H,W]


class ChannelAttention(nn.Module):
    """修正后的通道注意力模块"""

    def __init__(self):
        super().__init__()
        # 使用1x1卷积替代全连接层
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,3,1,1]
            nn.Conv2d(3, 1, kernel_size=1, bias=False),  # 3->1压缩
            nn.ReLU(),
            nn.Conv2d(1, 3, kernel_size=1, bias=False),  # 1->3恢复
            nn.Sigmoid()
        )

        # 修正初始化方式
        self._init_weights()

    def _init_weights(self):
        # 对3->1的卷积层初始化
        if hasattr(self.conv[1], 'weight'):
            weight = self.conv[1].weight  # shape: [1,3,1,1]
            nn.init.constant_(weight, 1 / 3)  # 平均初始化

        # 对1->3的卷积层初始化
        if hasattr(self.conv[3], 'weight'):
            weight = self.conv[3].weight  # shape: [3,1,1,1]
            nn.init.constant_(weight, 1.0)  # 单位初始化

    def forward(self, x):
        return self.conv(x)  # 输出 [B,3,1,1]


class LightDAM(nn.Module):
    """最终优化版双注意力模块（保持原始计算流程）"""

    def __init__(self):
        super().__init__()
        self.spatial_attn = SpatialAttention()
        self.channel_attn = ChannelAttention()

    def forward(self, x):
        # 输入检查（强制RGB输入）
        assert x.size(1) == 3, f"输入必须为3通道RGB，当前通道数: {x.size(1)}"

        # 1. 计算空间注意力 [B,1,H,W]
        spatial = self.spatial_attn(x)

        # 2. 计算通道注意力 [B,3,1,1]
        channel = self.channel_attn(x)

        #    a) 先Hadamard乘积（⊙）
        hadamard = spatial * channel  # 广播为[B,3,H,W]

        #    b) 再与输入X做逐元素乘（⊗）
        return x * hadamard  # 输出 [B,3,H,W]

