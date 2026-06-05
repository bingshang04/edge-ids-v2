"""
深度可分离时域卷积块 (Depthwise Separable TCN Block)

融合两条技术路线:
  论文5 (顾兆军等, 2025): 深度可分离卷积 + GAP 替代全连接层实现TCN轻量化
  论文2 (赵建, 姜伟, 2025): 并联1D卷积实现时空特征融合

设计要点:
  - 深度可分离卷积替代标准扩张因果卷积: 参数量减少约 8-9×
  - 并联1D常规卷积: 补充空间维度特征 (论文2并联思路)
  - 扩张因子指数增长: d=1,2,4,8,... 指数级扩大感受野
  - 权重归一化 + Dropout: 稳定训练，防止过拟合
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """
    一维深度可分离卷积: DWConv → PWConv

    计算量对比:
      标准1D卷积: C_in × C_out × K × L
      深度可分离:  C_in × K × L + C_in × C_out × L
      比例: ≈ 1/C_out + 1/K, C_out较大时约减少 K× 计算量

    论文来源:
      - 论文5 (顾兆军): 将深度可分离卷积引入TCN时间块
      - 论文3 (林硕): MobileNet思路在入侵检测中的应用
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ):
        super().__init__()

        if padding is None:
            # 因果卷积的零填充: 只在前端填充
            padding = (kernel_size - 1) * dilation

        # 深度卷积 (Depthwise): 每个输入通道独立卷积
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=in_channels,  # 关键: 分组数=输入通道数
            bias=bias,
        )

        # 逐点卷积 (Pointwise): 1×1卷积融合通道间信息
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)
        Returns:
            y: (B, C_out, L)
        """
        return self.pointwise(self.depthwise(x))


class StandardConv1dBlock(nn.Module):
    """标准1D卷积块 — 用于并联分支提取空间特征"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # same padding

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DSResidualTCNBlock(nn.Module):
    """
    深度可分离时域残差收缩块 (DS-TCN Block with optional SE Threshold)

    结构:
      ┌─ 深度可分离扩张因果卷积 ── WeightNorm + ReLU + Dropout
      │   (Depthwise Dilated Conv → Pointwise Conv)
      │
      ├─ + 标准1D卷积 ── BN + ReLU (论文2并联时空特征融合)
      │
      ├─ + SE自适应软阈值 (可选, 论文2冗余特征消除)
      │
      └─ + 残差连接 (1×1 Conv 维度匹配)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        use_se_threshold: bool = True,
        se_reduction: int = 4,
        use_spatial_branch: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        # --- 时间分支: 深度可分离扩张因果卷积 ---
        padding = (kernel_size - 1) * dilation  # 因果填充

        self.ds_conv = DepthwiseSeparableConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.temporal_bn = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        # --- 空间分支: 标准1D卷积 (论文2并联) ---
        self.use_spatial_branch = use_spatial_branch
        if use_spatial_branch:
            self.spatial_conv = StandardConv1dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )

        # --- SE自注意力软阈值 (论文2) ---
        self.use_se_threshold = use_se_threshold
        if use_se_threshold:
            from models.se_soft_threshold import SEAttentionThreshold

            self.se_threshold = SEAttentionThreshold(out_channels, se_reduction)

        # --- 残差连接 ---
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)
        Returns:
            y: (B, C_out, L_same) — 输出长度与输入一致
        """
        input_len = x.size(2)

        # 残差保存
        residual = self.residual_conv(x)

        # 时间分支: 深度可分离扩张因果卷积
        # 因果卷积在前端填充 (k-1)*d 个零，输出长度为 L + (k-1)*d
        out_temporal = self.ds_conv(x)
        # 裁剪到与输入相同长度 (截断后端多余部分，保持因果性)
        if out_temporal.size(2) > input_len:
            out_temporal = out_temporal[:, :, :input_len]
        out_temporal = self.temporal_bn(out_temporal)
        out_temporal = self.relu1(out_temporal)
        out_temporal = self.dropout1(out_temporal)

        # 空间分支: 并联融合 (论文2)
        if self.use_spatial_branch:
            spatial = self.spatial_conv(x)
            # 确保长度一致后相加
            if spatial.size(2) != out_temporal.size(2):
                spatial = spatial[:, :, :out_temporal.size(2)]
            out = out_temporal + spatial
        else:
            out = out_temporal

        # 自适应软阈值 (论文2 DRSM-CW)
        if self.use_se_threshold:
            out = self.se_threshold(out)

        # 残差连接
        if out.size(2) != residual.size(2):
            residual = residual[:, :, :out.size(2)]

        out = out + residual
        out = self.relu_out(out)

        return out
