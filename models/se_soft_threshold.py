"""
SE-Net 自适应软阈值 — 通道级冗余特征消除

参考论文: 赵建, 姜伟. 融合改进TCN与DRSN的IoT入侵检测模型. 小型微型计算机系统, 2025(2).
基于: DRSN-CW (Deep Residual Shrinkage Network with Channel-wise thresholds)

核心机制:
  1. 全局平均池化 → 压缩特征图 → 标量
  2. FC → ReLU → FC → Sigmoid → 缩放因子 α_c
  3. 阈值 τ_c = α_c × mean(|X_c|)
  4. 软阈值化: Y_c = sign(X_c) × max(|X_c| - τ_c, 0)

优势: 无需专家知识，自动为每个通道学习一个独立的软阈值
"""

import torch
import torch.nn as nn


class SEAttentionThreshold(nn.Module):
    """
    基于 SE-Net 注意力机制的通道级自适应软阈值

    Args:
        channels: 输入通道数
        reduction: SE模块的降维比率 (默认4，原论文用16，小模型用更小的降维比)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        reduced_channels = max(1, channels // reduction)

        # SE-Net 注意力分支: 压缩 → 激励
        self.se = nn.Sequential(
            # 全局平均池化: (B,C,L) → (B,C)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # 压缩
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            # 激励
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),  # 输出 (0,1) 缩放因子
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图, shape=(B, C, L)
        Returns:
            y: 软阈值化后的特征图, shape=(B, C, L)
        """
        B, C, L = x.shape

        # 1. SE分支计算每通道缩放因子 α
        alpha = self.se(x)  # (B, C)

        # 2. 计算每通道的绝对值均值作为基准阈值
        # mean(|X_c|) 在时间维度上做平均
        abs_mean = x.abs().mean(dim=2)  # (B, C)

        # 3. 自适应阈值: τ_c = α_c × mean(|X_c|)
        thresholds = alpha * abs_mean  # (B, C)

        # 4. 软阈值化: Y = sign(X) × max(|X| - τ, 0)
        # 将阈值广播到时间维度
        thresholds = thresholds.unsqueeze(2)  # (B, C, 1)

        # 软阈值化函数
        y = torch.sign(x) * torch.relu(x.abs() - thresholds)

        return y

    def get_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """返回当前输入的每通道阈值，用于分析和可视化"""
        alpha = self.se(x)  # (B, C)
        abs_mean = x.abs().mean(dim=2)  # (B, C)
        return alpha * abs_mean  # (B, C)
