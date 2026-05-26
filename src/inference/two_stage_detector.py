"""
两步级联推理器（Two-Stage Detector）
Model_A 二分类 → 若为攻击 → Model_B 九分类
"""
import numpy as np
from collections import deque
from typing import Dict, Any

from .result import DetectionResult


class TwoStageDetector:
    """两步级联推理包装器，接口与 IDSDetector 完全一致"""

    def __init__(self, model_a, model_b):
        """
        Args:
            model_a: 二分类检测器（Normal vs Attack）
            model_b: 九分类检测器（8种攻击子类 + Normal）
        """
        self.model_a = model_a
        self.model_b = model_b

        # 统计（合并两个模型）
        self._detection_count = 0
        self._attack_count = 0

    def predict(self, features: np.ndarray) -> DetectionResult:
        """级联推理：Model_A 判攻击后才调 Model_B"""
        result_a = self.model_a.predict(features)
        self._detection_count += 1

        if not result_a.is_attack():
            return result_a  # Normal，直接返回

        # 攻击 → 细分类
        result_b = self.model_b.predict(features)
        self._attack_count += 1

        # 如果 Model_B 也判 Normal（边界情况），使用 Model_B 的结果
        return result_b

    def predict_batch(self, features_list):
        return [self.predict(f) for f in features_list]

    def reset_buffer(self):
        self.model_a.reset_buffer()
        self.model_b.reset_buffer()

    def get_stats(self) -> Dict[str, Any]:
        stats_a = self.model_a.get_stats()
        stats_b = self.model_b.get_stats()
        return {
            'total_detections': self._detection_count,
            'attack_count': self._attack_count,
            'model_a': stats_a,
            'model_b': stats_b,
            'avg_latency_ms': stats_a.get('avg_latency_ms', 0),
            'max_latency_ms': max(
                stats_a.get('max_latency_ms', 0),
                stats_b.get('max_latency_ms', 0),
            ),
        }
