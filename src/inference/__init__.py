"""
Edge-IDS 推理检测模块
"""
from .detector import IDSDetector, DetectionResult, create_detector

__all__ = ['IDSDetector', 'DetectionResult', 'create_detector']
