"""
ECA (Efficient Channel Attention) 轻量级通道注意力模块
参考文献: Wang Q, Wu B, Zhu P, et al. ECA-Net: Efficient Channel Attention
         for Deep Convolutional Neural Networks[C]. CVPR, 2020.
"""
import torch
import torch.nn as nn


class ECALayer(nn.Module):
    """
    一维通道注意力模块

    通过自适应大小的1D卷积捕获跨通道交互，几乎不增加计算量。
    参数量: 仅 k 个 (~3-5个)，标准SENet需要 C²/r 个参数。
    """

    def __init__(self, channels: int, gamma: int = 2, bias: int = 1):
        """
        Args:
            channels: 输入通道数
            gamma: 用于自适应计算卷积核大小的系数
            bias: 用于自适应计算卷积核大小的偏移
        """
        super().__init__()

        # 自适应计算卷积核大小
        # k = |log2(C)/γ + b/γ|odd，确保为奇数
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + bias) / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        kernel_size = max(3, kernel_size)  # 最小为3

        self.kernel_size = kernel_size

        # 只用1个1D卷积实现通道注意力（无需全连接层）
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)

        Returns:
            加权后的特征图，shape 不变
        """
        # 全局平均池化 → (batch, channels, 1)
        y = x.mean(dim=-1, keepdim=True)

        # 转置为 (batch, 1, channels) 做1D卷积
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)

        # Sigmoid激活得到注意力权重
        y = self.sigmoid(y)

        # 通道加权
        return x * y

    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}'
