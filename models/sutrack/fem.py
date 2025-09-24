import torch
import torch.nn as nn
import math

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积替代标准分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FEM(nn.Module):
    """鲁棒的FEM模块，支持非常规形状输入"""

    def __init__(self, in_channels=384, out_channels=384):
        super().__init__()
        self.in_channels = in_channels

        # 分支1: 1x1标准卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 分支2: 深度可分离卷积替代
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels // 4, (1, 3), (0, 1)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels // 4, out_channels // 4, (3, 1), (1, 0)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 分支3: 轻量级空洞卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3,
                      padding=1, dilation=1),  # 保持局部连续性
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3,
                      padding=2, dilation=2),  # 扩大感受野
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 分支4: 标准3x3卷积
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 动态形状适配层
        self.shape_adapter = nn.Linear(246, 225)  # 246 -> 15x15=225

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # def forward(self, x):
    #     # 处理多尺度输入
    #     if isinstance(x, (list, tuple)):
    #         x = x[-1]
    #
    #     B, N, C = x.shape
    #
    #     # 动态形状适配 (246 -> 225)
    #     if N == 246:
    #         x = self.shape_adapter(x.transpose(1, 2)).transpose(1, 2)
    #         N = 225
    #         H = W = 15
    #     else:
    #         H = W = int(math.sqrt(N))
    #
    #     # 转换为空间特征
    #     x = x.transpose(1, 2).reshape(B, C, H, W)
    #
    #     # 并行处理四个分支
    #     b1 = self.branch1(x)
    #     b2 = self.branch2(x)
    #     b3 = self.branch3(x)
    #     b4 = self.branch4(x)
    #
    #     # 特征融合
    #     fused = self.fusion(torch.cat([b1, b2, b3, b4], dim=1))
    #     return fused.flatten(2).transpose(1, 2)
    def forward(self, x):
        # 保持原始输入处理
        if isinstance(x, (list, tuple)):
            x = x[-1]

        B, N, C = x.shape  # 获取输入形状

        # 自动计算合法的H和W（关键修改）
        H = W = int(math.sqrt(N))  # 计算理论尺寸
        if H * W != N:  # 如果无法整除
            # 取最接近的较小平方数
            H = W = int(math.sqrt(N))
            N = H * W
            x = x[:, :N, :]  # 裁剪多余元素

        # 继续原有操作
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # 保持原有分支处理
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 保持原有融合输出
        fused = self.fusion(torch.cat([b1, b2, b3, b4], dim=1))
        return fused.flatten(2).transpose(1, 2)

