import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .constants import (
    PROJECT_NAME, DEFAULT_MODEL_PATH, DEFAULT_LOG_DIR,
    PLATFORM_DEFAULTS, FEATURE_CONFIG, LOG_CONFIG,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 48
    num_classes: int = 10
    num_channels: list = field(default_factory=lambda: [128, 256, 256])
    kernel_size: int = 5
    dropout: float = 0.3
    use_eca: bool = True
    model_path: str = DEFAULT_MODEL_PATH


@dataclass
class CaptureConfig:
    """数据包捕获配置"""
    interface: str = 'auto'
    bpf_filter: str = 'ip'
    buffer_size: int = 65536
    promiscuous: bool = True
    queue_size: int = 10000


@dataclass
class FeatureConfig:
    """特征提取配置"""
    flow_timeout: float = 120.0
    max_flows: int = 10000
    feature_dim: int = 48
    packet_history_size: int = 1000
    iat_history_size: int = 100


@dataclass
class InferenceConfig:
    """推理配置"""
    confidence_threshold: float = 0.5
    alert_threshold: float = 0.8
    sequence_length: int = 10
    batch_size: int = 64
    buffer_size: int = 100
    scaler_path: Optional[str] = None


@dataclass
class WebConfig:
    """Web服务器配置"""
    host: str = '0.0.0.0'
    port: int = 8080
    debug: bool = False


@dataclass
class LogConfig:
    """日志配置"""
    level: str = 'INFO'
    format: str = LOG_CONFIG['format']
    date_format: str = LOG_CONFIG['date_format']
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    log_dir: str = DEFAULT_LOG_DIR


class Settings:
    """统一配置管理（单例模式）"""

    _instance: Optional['Settings'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return

        self._initialized = True
        self._config_path = config_path
        self._platform_type = self._detect_platform()

        self.model = ModelConfig()
        self.capture = CaptureConfig()
        self.feature = FeatureConfig()
        self.inference = InferenceConfig()
        self.web = WebConfig()
        self.log = LogConfig()
        self.platform: Dict[str, Any] = {}

        self._load_config()

    def _detect_platform(self) -> str:
        import platform
        sys_platform = platform.system().lower()
        machine = platform.machine().lower()

        if sys_platform == 'windows':
            return 'windows'
        elif 'arm' in machine or 'aarch64' in machine:
            return 'raspberry_pi'
        elif sys_platform == 'linux':
            return 'x86_pc'
        return 'x86_pc'

    def _load_config(self):
        self._apply_platform_defaults()

        if self._config_path and os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                if config_data:
                    self._apply_yaml_config(config_data)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")

        self._apply_env_overrides()

    def _apply_platform_defaults(self):
        defaults = PLATFORM_DEFAULTS.get(self._platform_type, PLATFORM_DEFAULTS['x86_pc'])
        self.platform = {'type': self._platform_type, **defaults}
        self.inference.sequence_length = defaults['sequence_length']
        self.inference.batch_size = defaults['batch_size']
        # 优先使用显式的 num_channels，否则用 hidden_dim * num_layers 计算
        if 'num_channels' in defaults:
            self.model.num_channels = defaults['num_channels']
        else:
            self.model.num_channels = [defaults['hidden_dim']] * defaults['num_layers']
        self.model.kernel_size = defaults['kernel_size']
        self.model.dropout = defaults['dropout']

    def _apply_yaml_config(self, config_data: Dict[str, Any]):
        for section, config_obj in [
            ('model', self.model), ('capture', self.capture),
            ('features', self.feature), ('inference', self.inference),
            ('web', self.web), ('logging', self.log),
        ]:
            if section in config_data:
                for key, value in config_data[section].items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)

    def _apply_env_overrides(self):
        env_mappings = {
            'EDGE_IDS_INTERFACE': ('capture', 'interface'),
            'EDGE_IDS_MODEL_PATH': ('model', 'model_path'),
            'EDGE_IDS_WEB_PORT': ('web', 'port'),
            'EDGE_IDS_LOG_LEVEL': ('log', 'level'),
        }
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                config_obj = getattr(self, section)
                try:
                    current = getattr(config_obj, key)
                    if isinstance(current, int):
                        value = int(value)
                    elif isinstance(current, float):
                        value = float(value)
                except (ValueError, AttributeError):
                    pass
                setattr(config_obj, key, value)

    @property
    def platform_type(self) -> str:
        return self._platform_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            'platform': self.platform,
            'model': self.model.__dict__,
            'capture': self.capture.__dict__,
            'feature': self.feature.__dict__,
            'inference': self.inference.__dict__,
            'web': self.web.__dict__,
            'log': self.log.__dict__,
        }


def get_settings(config_path: Optional[str] = None) -> Settings:
    return Settings(config_path)
