"""
训练配置文件

集中管理所有超参数，方便实验管理和消融实验
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型架构配置"""

    # 输入维度
    input_dim: int = 49  # UNSW-NB15 特征维度
    num_classes: int = 10  # 9攻击 + 1正常

    # TCN 配置
    tcn_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4])
    kernel_size: int = 3
    dropout: float = 0.40  # 加大防过拟合

    # 轻量化开关
    use_se_threshold: bool = True  # SE自适应软阈值 (论文2)
    use_spatial_branch: bool = True  # 并联1D空间分支 (论文2)
    use_gap: bool = True  # GAP替代全连接层 (论文5)

    # 模型规模预设: 'tiny' | 'small' | 'medium'
    model_size: str = "small"


@dataclass
class TrainConfig:
    """训练配置"""

    # 优化器
    optimizer: str = "adam"
    learning_rate: float = 0.001  # 初始学习率
    lr_decay: float = 0.1  # 学习率衰减因子
    lr_decay_epochs: int = 30  # 衰减触发epoch (论文2: 前30 epoch 0.001, 后20 epoch 0.0001)
    weight_decay: float = 1e-4  # L2正则化 (防过拟合)

    # 训练
    epochs: int = 50
    batch_size: int = 64  # 论文2: batch=64
    early_stop_patience: int = 15  # 早停耐心值 (OneCycleLR需要更长探索)

    # Focal Loss (论文2)
    focal_gamma: float = 3.0  # 聚焦参数 (增大以强化稀有类)
    use_focal_loss: bool = True
    use_weighted_sampler: bool = True  # 加权采样过采样稀有类

    # 数据
    train_split: float = 0.7
    val_split: float = 0.1  # 测试集=1-train_split-val_split
    shuffle: bool = True
    num_workers: int = 2

    # 序列化窗口
    sequence_length: int = 100

    # 设备
    device: str = "cuda"  # 'cuda' | 'cpu'

    # 日志
    log_interval: int = 10  # 每N个batch打印一次日志
    save_dir: str = "./checkpoints"


@dataclass
class DataConfig:
    """数据配置"""

    # UNSW-NB15
    unsw_nb15_path: str = "./data/raw/UNSW-NB15/"

    # 预处理
    normalize: str = "zscore"  # 'zscore' | 'minmax' | 'none'
    handle_missing: str = "drop"  # 'drop' | 'fill' | 'drop_col'

    # 类别不平衡处理
    imbalance_method: str = "focal_loss"  # 'focal_loss' | 'smote' | 'class_weight' | 'none'

    # 序列构建
    window_size: int = 100
    stride: int = 1
    min_sequence_length: int = 10


@dataclass
class Config:
    """总配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


# --- 预设配置 ---

def get_tiny_config() -> Config:
    """树莓派5极致轻量部署配置"""
    config = Config()
    config.model.model_size = "tiny"
    config.model.tcn_channels = [32, 64, 128]
    config.model.dilations = [1, 2, 4]
    config.model.use_gap = True
    config.model.use_se_threshold = True
    config.train.batch_size = 32
    return config


def get_small_config() -> Config:
    """推荐均衡配置 (默认)"""
    return Config()


def get_medium_config() -> Config:
    """GPU高性能配置 (对比实验)"""
    config = Config()
    config.model.model_size = "medium"
    config.model.tcn_channels = [128, 256, 512, 512]
    config.model.dilations = [1, 2, 4, 8]
    config.model.use_gap = True
    config.model.dropout = 0.4
    config.train.batch_size = 64
    config.train.device = "cuda"
    return config
