"""
训练配置文件 (v2 重构)

简化配置 — 主要超参数在 train.py 常量区，此处保留模型和数据配置
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型架构配置"""

    # 输入维度 (由数据自动推断)
    input_dim: int = 48
    num_classes: int = 10  # 10 类（DoS/Exploits 独立）

    # TCN 配置
    tcn_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 256])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size: int = 5
    dropout: float = 0.3

    # DS-TCN 特征开关
    use_se_threshold: bool = True
    use_spatial_branch: bool = True
    use_gap: bool = True

    # 模型规模: 'tiny' | 'small' | 'medium'
    model_size: str = "small"


@dataclass
class TrainConfig:
    """训练配置"""

    # 优化器
    learning_rate: float = 0.001
    weight_decay: float = 5e-5

    # 训练
    epochs: int = 30
    batch_size: int = 64
    early_stop_patience: int = 8

    # 窗口
    sequence_length: int = 10

    # Label Smoothing (替代 FocalLoss — SMOTE 已处理不平衡)
    label_smoothing: float = 0.1

    # 设备
    device: str = "cuda"

    # 日志
    save_dir: str = "./checkpoints"


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --- 预设配置 ---

def get_tiny_config() -> Config:
    """树莓派5轻量配置"""
    config = Config()
    config.model.model_size = "tiny"
    config.model.tcn_channels = [32, 64, 128]
    config.model.dilations = [1, 2, 4]
    config.train.batch_size = 32
    return config


def get_small_config() -> Config:
    """推荐均衡配置 (默认)"""
    config = Config()
    config.model.model_size = "small"
    config.model.tcn_channels = [64, 128, 256, 256]
    config.model.dilations = [1, 2, 4, 8]
    return config


def get_medium_config() -> Config:
    """GPU高性能配置"""
    config = Config()
    config.model.model_size = "medium"
    config.model.tcn_channels = [128, 256, 512, 512]
    config.model.dilations = [1, 2, 4, 8]
    config.train.batch_size = 64
    return config
