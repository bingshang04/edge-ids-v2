"""
Focal Loss — 解决入侵检测数据类别不平衡问题

参考论文: 赵建, 姜伟. 融合改进TCN与DRSN的IoT入侵检测模型. 小型微型计算机系统, 2025(2).
公式: FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

其中:
  γ = 2      — 调节因子，降低易分类样本权重，使模型关注难分类样本
  α_t        — 类别平衡因子，给少数类分配更高权重
  p_t        — 分类置信度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    多分类 Focal Loss

    Args:
        alpha: 类别权重，shape=(num_classes,) 或 None (自动按样本比例计算)
        gamma: 聚焦参数，默认 2.0 (论文推荐值)
        reduction: 'mean' | 'sum' | 'none'
        dynamic_alpha: 是否在训练中动态更新 alpha (基于每个batch的类别分布)
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        dynamic_alpha: bool = False,
    ):
        super().__init__()
        self.alpha = alpha  # 静态权重 (可预先计算好传入)
        self.gamma = gamma
        self.reduction = reduction
        self.dynamic_alpha = dynamic_alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型 logits, shape=(N, C)
            targets: 真实标签, shape=(N,) 整型类别索引
        Returns:
            loss: 标量或逐样本损失
        """
        num_classes = inputs.size(1)

        # 计算交叉熵损失 (不 reduction, 保留逐样本)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # (N,)

        # 计算 p_t: 模型对真实类别的预测概率
        # p_t = softmax(inputs) 在真实类别上的值
        p = F.softmax(inputs, dim=1)  # (N, C)
        # 取出真实类别对应的概率
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)

        # 聚焦权重: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        # α_t: 类别平衡权重
        if self.alpha is not None:
            # 使用预计算的静态 alpha
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]  # (N,)
        elif self.dynamic_alpha:
            # 动态计算：基于当前 batch 的类别分布
            # α_t = 1 - n_t / sum_n  (论文式4)
            class_counts = torch.bincount(targets, minlength=num_classes).float()
            total = class_counts.sum()
            # 避免除零
            class_weights = 1.0 - class_counts / (total + 1e-8)
            alpha_t = class_weights[targets]
        else:
            alpha_t = 1.0

        # Focal Loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def compute_class_weights(
        labels: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """
        根据全量训练标签预计算类别权重

        使用逆平方根频率 (比论文2公式4更适合极端不平衡):
          α_t = 1 / sqrt(n_t)
        对于 Normal(10830个) vs Worms(17个):
          逆频率比 ≈ 25× (vs 原公式的 1.5×)

        Args:
            labels: 训练集所有标签, shape=(N,)
            num_classes: 类别总数
        Returns:
            alpha: 类别权重, shape=(num_classes,)
        """
        class_counts = torch.bincount(labels, minlength=num_classes).float()
        # 逆平方根频率: 比原公式 1-n_t/N 对稀有类更激进
        # 1/sqrt(10830)=0.0096 vs 1/sqrt(17)=0.2425 → 25× 差异
        alpha = 1.0 / torch.sqrt(class_counts + 1)  # +1防除零
        # 归一化使平均权重为1
        alpha = alpha / alpha.mean()
        return alpha
