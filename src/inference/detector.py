"""
入侵检测推理模块（多分类 ECA-TCN，使用训练时保存的 Scaler 和 LabelEncoder）
"""
import time
import warnings
import joblib
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..utils.logger import LoggerMixin
from ..utils.exceptions import ModelError, InferenceError
from ..models.tcn_model import TCN

# 攻击类型 ID → 名称映射（与训练端 LabelEncoder 一致）
# 0: Normal, 1: Analysis, 2: Backdoor, 3: DoS, 4: Exploits,
# 5: Fuzzers, 6: Generic, 7: Reconnaissance, 8: Shellcode, 9: Worms
ATTACK_TYPE_NAMES: Dict[int, str] = {
    0: 'Normal',
    1: 'Analysis',
    2: 'Backdoor',
    3: 'DoS',
    4: 'Exploits',
    5: 'Fuzzers',
    6: 'Generic',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms',
}

# 攻击类型 → 危险等级映射
ATTACK_DANGER_LEVELS: Dict[int, str] = {
    0: '无',         # Normal
    1: '中危',       # Analysis
    2: '严重',       # Backdoor
    3: '严重',       # DoS
    4: '严重',       # Exploits
    5: '高危',       # Fuzzers
    6: '高危',       # Generic
    7: '中危',       # Reconnaissance
    8: '严重',       # Shellcode
    9: '严重',       # Worms
}


@dataclass
class DetectionResult:
    """多分类检测结果"""
    prediction: int          # 类别 ID (0-9)
    confidence: float        # 预测置信度（softmax 最大值）
    probability: float       # 攻击概率（所有非 Normal 类的 softmax 和）
    latency_ms: float
    timestamp: float
    attack_type_id: int = 0            # 攻击类型 ID
    attack_type_name: str = 'Normal'   # 攻击类型名称
    danger_level: str = '无'           # 危险等级

    def is_attack(self, threshold: float = 0.5) -> bool:
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


class IDSDetector(LoggerMixin):
    """实时入侵检测器"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_dim: int = 48,
        num_classes: int = 10,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
        sequence_length: int = 10,
        confidence_threshold: float = 0.5,
        alert_threshold: float = 0.8,
        use_eca: bool = True,
        use_quantization: bool = False,
        scaler_path: Optional[str] = None,
        le_attack_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.alert_threshold = alert_threshold

        # 设备
        self.device = self._setup_device(device)

        # 模型
        self.model = self._build_model(
            model_path, num_channels, kernel_size, dropout, use_eca, use_quantization
        )

        # 加载 Scaler（训练时保存的标准化器）
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            # 清除特征名，避免推理时 DataFrame vs ndarray 警告
            if hasattr(self.scaler, 'feature_names_in_'):
                self.scaler.feature_names_in_ = None
            self.logger.info(f"已加载 Scaler: {scaler_path}")

        # 加载 attack_cat LabelEncoder（训练时保存的）
        self._le_attack = None
        if le_attack_path and Path(le_attack_path).exists():
            self._le_attack = joblib.load(le_attack_path)
            self.logger.info(f"已加载 attack_cat LabelEncoder: {le_attack_path} ({len(self._le_attack.classes_)} 类)")

        # 序列缓冲区
        self._feature_buffer: deque = deque(maxlen=sequence_length)

        # 统计
        self._inference_times: deque = deque(maxlen=100)
        self._detection_count = 0
        self._attack_count = 0

    def _setup_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _build_model(self, model_path, num_channels, kernel_size, dropout, use_eca, use_quantization):
        model = TCN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            num_channels=num_channels or [128, 256, 256],
            kernel_size=kernel_size,
            dropout=dropout,
            use_eca=use_eca,
        )

        if model_path and Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.logger.info(f"已加载模型: {model_path}")
            except Exception as e:
                self.logger.warning(f"模型加载失败: {e}，使用未训练模型")

        if use_quantization:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, features: np.ndarray) -> torch.Tensor:
        """使用训练时的 Scaler 进行标准化"""
        if len(features) != self.input_dim:
            raise InferenceError(f"特征维度不匹配: 期望 {self.input_dim}, 实际 {len(features)}")

        # 使用训练时保存的 Scaler
        if self.scaler is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
        else:
            # 降级方案：Z-score 归一化
            mean, std = features.mean(), features.std()
            features = (features - mean) / (std + 1e-8)

        self._feature_buffer.append(features)

        # 填充序列
        while len(self._feature_buffer) < self.sequence_length:
            self._feature_buffer.append(features)

        sequence = np.array(list(self._feature_buffer)[-self.sequence_length:])
        return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

    def predict(self, features: np.ndarray) -> DetectionResult:
        start = time.time()
        try:
            tensor = self._preprocess(features)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # 多分类: 取 softmax 最大值作为预测类别
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            # 攻击概率 = 所有非 Normal 类（id != 0）的 softmax 之和
            attack_prob = float(np.sum(probs[1:])) if len(probs) > 1 else 0.0

            # 攻击类型信息
            attack_type_id = pred_class
            attack_type_name = ATTACK_TYPE_NAMES.get(pred_class, f'Unknown-{pred_class}')
            danger_level = ATTACK_DANGER_LEVELS.get(pred_class, '未知')

            latency_ms = (time.time() - start) * 1000
            self._inference_times.append(latency_ms)
            self._detection_count += 1
            if pred_class != 0:
                self._attack_count += 1

            return DetectionResult(
                prediction=pred_class,
                confidence=confidence,
                probability=attack_prob,
                latency_ms=latency_ms,
                timestamp=time.time(),
                attack_type_id=attack_type_id,
                attack_type_name=attack_type_name,
                danger_level=danger_level,
            )
        except Exception as e:
            self.logger.error(f"推理错误: {e}")
            raise InferenceError(f"推理失败: {e}")

    def predict_batch(self, features_list: List[np.ndarray]) -> List[DetectionResult]:
        return [self.predict(f) for f in features_list]

    def reset_buffer(self):
        self._feature_buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        times = list(self._inference_times)
        if not times:
            return {'total_detections': 0, 'attack_count': 0}
        return {
            'total_detections': self._detection_count,
            'attack_count': self._attack_count,
            'avg_latency_ms': round(np.mean(times), 2),
            'max_latency_ms': round(np.max(times), 2),
            'p95_latency_ms': round(np.percentile(times, 95), 2) if len(times) >= 20 else round(np.max(times), 2),
        }

    def save_model(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def export_onnx(self, path: str):
        dummy = torch.randn(1, self.sequence_length, self.input_dim).to(self.device)
        torch.onnx.export(self.model, dummy, path,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})


def create_detector(config=None) -> IDSDetector:
    if config:
        project_root = Path(__file__).parent.parent.parent
        model_dir = project_root / 'data' / 'models'
        return IDSDetector(
            model_path=config.model.model_path,
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes if hasattr(config.model, 'num_classes') else 10,
            num_channels=config.model.num_channels,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout,
            use_eca=config.model.use_eca if hasattr(config.model, 'use_eca') else True,
            sequence_length=config.inference.sequence_length,
            confidence_threshold=config.inference.confidence_threshold,
            alert_threshold=config.inference.alert_threshold,
            scaler_path=config.inference.scaler_path,
            le_attack_path=str(model_dir / 'le_attack_cat.joblib'),
        )
    return IDSDetector()
