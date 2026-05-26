from enum import Enum
from typing import Dict, Any

# 项目信息
PROJECT_NAME = "Edge-IDS"
PROJECT_VERSION = "2.0.0"
PROJECT_DESCRIPTION = "基于 ECA-TCN 的轻量级边缘入侵检测系统"

# 默认路径
DEFAULT_MODEL_PATH = "data/models/tcn_model_3.0.pth"
DEFAULT_SCALER_PATH = "data/models/unsw_scaler3.0.joblib"
DEFAULT_LOG_DIR = "logs"

# 特征配置
FEATURE_CONFIG = {
    'flow_timeout': 120.0,
    'max_flows': 10000,
    'feature_dim': 48,  # 统一为48维
    'packet_history_size': 1000,
    'iat_history_size': 100,
}

# 平台默认配置
PLATFORM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'raspberry_pi': {
        'batch_size': 16,
        'sequence_length': 10,
        'num_channels': [64, 128],
        'use_tflite': True,
        'tflite_model_path': 'data/models/tcn_model.tflite',
        'tflite_model_a_path': 'data/models/tcn_model_binary.tflite',
        'tflite_model_b_path': 'data/models/tcn_model_9class.tflite',
        'tflite_num_threads': 4,
        'quantization': 'INT8',
        'max_memory_mb': 2048,
        'inference_threads': 4,
        'learning_rate': 0.001,
        'kernel_size': 3,
        'dropout': 0.2,
    },
    'x86_pc': {
        'batch_size': 64,
        'sequence_length': 10,
        'hidden_dim': 128,
        'num_layers': 3,
        'use_tflite': False,
        'quantization': 'FP32',
        'max_memory_mb': 8192,
        'inference_threads': 8,
        'learning_rate': 0.001,
        'kernel_size': 5,
        'dropout': 0.3,
    },
    'windows': {
        'batch_size': 64,
        'sequence_length': 10,
        'hidden_dim': 128,
        'num_layers': 3,
        'use_tflite': False,
        'quantization': 'FP32',
        'max_memory_mb': 8192,
        'inference_threads': 8,
        'learning_rate': 0.001,
        'kernel_size': 5,
        'dropout': 0.3,
    },
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'max_bytes': 10 * 1024 * 1024,
    'backup_count': 5,
}
