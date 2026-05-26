"""
Edge-IDS 推理检测模块
按平台自动选择推理后端（TFLite / PyTorch）
"""
from .result import DetectionResult       # 共享数据模型
from .detector import IDSDetector          # PyTorch 推理器
from .tflite_detector import TFLiteDetector  # TFLite 推理器
from .two_stage_detector import TwoStageDetector  # 两步级联推理器
from .factory import create_detector       # 工厂函数（自动选择后端）

__all__ = [
    'IDSDetector',
    'TFLiteDetector',
    'TwoStageDetector',
    'DetectionResult',
    'create_detector',
]
