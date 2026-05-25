"""
共享的检测结果数据模型（无 torch/tflite 依赖）
供 IDSDetector 和 TFLiteDetector 共同使用
"""
import time
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DetectionResult:
    """多分类检测结果（框架无关）"""
    prediction: int           # 类别 ID (0-9)
    confidence: float         # 预测置信度（softmax 最大值）
    probability: float        # 攻击概率（所有非 Normal 类的 softmax 和）
    latency_ms: float
    timestamp: float
    attack_type_id: int = 0            # 攻击类型 ID
    attack_type_name: str = 'Normal'   # 攻击类型名称
    danger_level: str = '无'           # 危险等级

    def is_attack(self, threshold: float = 0.5) -> bool:
        """判断是否为攻击（prediction != Normal 且置信度达标）"""
        return self.prediction != 0 and self.confidence >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction': self.prediction,
            'confidence': round(self.confidence, 4),
            'probability': round(self.probability, 4),
            'latency_ms': round(self.latency_ms, 2),
            'timestamp': self.timestamp,
            'is_attack': self.is_attack(),
            'attack_type_id': self.attack_type_id,
            'attack_type_name': self.attack_type_name,
            'danger_level': self.danger_level,
        }
