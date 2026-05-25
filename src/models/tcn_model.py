"""
改进版 TCN 模型 — 集成 ECA 通道注意力机制
"""
import torch
import torch.nn as nn
from typing import List, Optional

from .eca import ECALayer


class Chomp1d(nn.Module):
    """1D 因果卷积裁剪层"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

    def extra_repr(self) -> str:
        return f'chomp_size={self.chomp_size}'


class TemporalBlock(nn.Module):
    """
    时序卷积块（含 ECA 注意力）

    两个扩张卷积层 + ECA 通道注意力 + 残差连接
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        use_eca: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # ECA 注意力（加在残差之前）
        self.eca = ECALayer(n_outputs) if use_eca else nn.Identity()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = self.eca(out)  # ECA 通道注意力
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    时序卷积网络（TCN）+ ECA 注意力

    用于网络入侵检测的时序特征学习。
    改进点：每个 TemporalBlock 中集成 ECA 通道注意力机制。
    """

    def __init__(
        self,
        input_dim: int = 48,
        num_classes: int = 2,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
        use_eca: bool = True,
    ):
        """
        Args:
            input_dim: 输入特征维度
            num_classes: 输出类别数
            num_channels: 各层通道数
            kernel_size: 卷积核大小
            dropout: Dropout 概率
            use_eca: 是否使用 ECA 通道注意力
        """
        super().__init__()

        if num_channels is None:
            num_channels = [128, 256, 256]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.use_eca = use_eca

        # 构建 TCN 层
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation, padding=padding,
                dropout=dropout, use_eca=use_eca
            ))

        self.network = nn.Sequential(*layers)

        # 全局平均池化 + Dropout（防止过拟合）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout_out = nn.Dropout(dropout)

        # 分类器
        self.classifier = nn.Linear(num_channels[-1], num_classes)

        # 统计参数量
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            logits: (batch, num_classes)
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        features = self.network(x)
        pooled = self.global_pool(features).squeeze(-1)
        pooled = self.dropout_out(pooled)
        logits = self.classifier(pooled)
        return logits

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """获取池化后的特征表示"""
        x = x.transpose(1, 2)
        features = self.network(x)
        pooled = self.global_pool(features).squeeze(-1)
        return pooled

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(x), dim=1)

    def get_model_info(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'use_eca': self.use_eca,
            'num_parameters': self.num_params,
            'model_size_mb': self.num_params * 4 / (1024 * 1024),
        }
