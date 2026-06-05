"""
DS-TCN-IDS — 完整深度可分离时域卷积网络入侵检测模型

技术来源整合:
  - 论文1 (Nazre et al. 2024): TCN残差块架构 + 空洞卷积感受野设计
  - 论文2 (赵建, 姜伟 2025): 并联1D卷积 + SE软阈值 + Focal Loss
  - 论文5 (顾兆军等 2025): 深度可分离卷积替代标准卷积 + GAP替代全连接层

架构概述:
  输入 (B, seq_len, features)
    → 1D 初始卷积 (features→64)
    → DS-TCN Block ×3 (d=1,2,4; filters=64,128,256)
    → 全局平均池化 (GAP)
    → Dropout → Dense(num_classes)
    → Softmax
"""

import torch
import torch.nn as nn

from models.ds_tcn import DSResidualTCNBlock


class DSTCNIDS(nn.Module):
    """
    DS-TCN-IDS 完整模型

    Args:
        input_dim: 输入特征维度 (UNSW-NB15: 49维)
        num_classes: 分类类别数 (UNSW-NB15: 10类 = 9攻击+1正常)
        tcn_channels: 各TCN块的输出通道数, 默认 [64, 128, 256]
        dilations: 各TCN块的扩张因子, 默认 [1, 2, 4]
        kernel_size: 卷积核大小
        dropout: Dropout比率
        use_se_threshold: 是否使用SE自适应软阈值 (论文2)
        use_spatial_branch: 是否使用并联1D空间分支 (论文2)
        use_gap: 是否使用GAP替代全连接层 (论文5), True=只有最终分类头, False=保留FC
    """

    def __init__(
        self,
        input_dim: int = 49,
        num_classes: int = 10,
        tcn_channels: list[int] | None = None,
        dilations: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_se_threshold: bool = True,
        use_spatial_branch: bool = True,
        use_gap: bool = True,
    ):
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [64, 128, 256]
        if dilations is None:
            dilations = [1, 2, 4]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_gap = use_gap
        self.use_se_threshold = use_se_threshold
        self.use_spatial_branch = use_spatial_branch

        # --- 初始1D卷积: 特征维度映射 ---
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_dim, tcn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(tcn_channels[0]),
            nn.ReLU(inplace=True),
        )

        # --- DS-TCN 残差块堆叠 ---
        tcn_blocks = []
        in_ch = tcn_channels[0]

        for out_ch, dilation in zip(tcn_channels, dilations):
            tcn_blocks.append(
                DSResidualTCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_se_threshold=use_se_threshold,
                    use_spatial_branch=use_spatial_branch,
                )
            )
            in_ch = out_ch

        self.tcn_blocks = nn.Sequential(*tcn_blocks)

        # --- 批归一化 + 激活 ---
        self.post_tcn = nn.Sequential(
            nn.BatchNorm1d(tcn_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # --- 全局平均池化 (论文5) 或 全连接层 ---
        final_dim = tcn_channels[-1]

        if use_gap:
            # GAP: 参数为0, 大幅减少参数量 (论文5)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(final_dim, num_classes)
        else:
            # 传统方案: Flatten → FC → Dropout → FC
            self.gap = None
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(dropout)
            # 需要 seq_len 才能计算 FC 输入维度, 这里用全局平均池化兜底
            self.fc = nn.Linear(final_dim, 64)
            self.classifier = nn.Linear(64, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """He Normal 初始化 (适配ReLU激活)"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, features) — 输入序列
        Returns:
            logits: (B, num_classes)
        """
        # 转置为 Conv1d 需要的格式: (B, features, seq_len)
        x = x.transpose(1, 2)  # (B, seq_len, features) → (B, features, seq_len)

        # 初始卷积
        x = self.init_conv(x)  # (B, 64, seq_len)

        # DS-TCN残差块
        x = self.tcn_blocks(x)  # (B, 256, seq_len)

        # 后处理
        x = self.post_tcn(x)  # (B, 256, seq_len)

        # 全局平均池化 或 传统FC
        if self.use_gap:
            x = self.gap(x)  # (B, 256, 1)
            x = x.squeeze(-1)  # (B, 256)
            x = self.dropout(x)
        else:
            # 用自适应池化降维后接FC
            pooled = torch.mean(x, dim=2)  # (B, 256)
            x = self.dropout(pooled)
            x = torch.relu(self.fc(x))

        # 分类头
        logits = self.classifier(x)  # (B, num_classes)

        return logits

    def get_model_size(self) -> tuple[int, float]:
        """获取模型参数量和估算体积"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        # 估算FP32模型大小 (字节)
        size_bytes = total_params * 4  # float32 = 4 bytes
        size_mb = size_bytes / (1024 * 1024)
        return trainable_params, size_mb


def create_model(
    input_dim: int = 49,
    num_classes: int = 10,
    model_size: str = "small",  # 'tiny', 'small', 'medium'
) -> DSTCNIDS:
    """
    工厂函数: 根据预设配置创建模型

    | 配置 | 通道数 | 扩张 | 适用场景 |
    |------|--------|------|---------|
    | tiny | [32,64,128] | [1,2,4] | 极致轻量 (树莓派5部署) |
    | small | [64,128,256] | [1,2,4] | 推荐均衡配置 |
    | medium | [128,256,512] | [1,2,4,8] | 高性能 (GPU部署) |
    """
    configs = {
        "tiny": {
            "tcn_channels": [32, 64, 128],
            "dilations": [1, 2, 4],
        },
        "small": {
            "tcn_channels": [64, 128, 256],
            "dilations": [1, 2, 4],
        },
        "medium": {
            "tcn_channels": [128, 256, 512, 512],
            "dilations": [1, 2, 4, 8],
        },
    }

    cfg = configs.get(model_size, configs["small"])
    return DSTCNIDS(
        input_dim=input_dim,
        num_classes=num_classes,
        tcn_channels=cfg["tcn_channels"],
        dilations=cfg["dilations"],
    )
