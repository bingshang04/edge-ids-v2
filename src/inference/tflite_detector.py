"""
TFLite 推理器（树莓派5 ARM64 部署用）
使用 XNNPACK delegate，支持 INT8 量化模型
接口与 IDSDetector 保持一致
"""
import time
import warnings
import logging
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# 尝试导入 TFLite 运行时（树莓派用 tflite-runtime，PC 用 tensorflow）
try:
    import tflite_runtime.interpreter as tflite_interpreter
    HAS_TFLITE = True
    TFLITE_BACKEND = 'tflite-runtime'
except ImportError:
    try:
        import tensorflow.lite as tflite_interpreter
        HAS_TFLITE = True
        TFLITE_BACKEND = 'tensorflow'
    except ImportError:
        HAS_TFLITE = False
        TFLITE_BACKEND = 'none'

from .result import DetectionResult  # 共享数据模型（无框架依赖）
from ..utils.logger import LoggerMixin
from ..utils.exceptions import ModelError, InferenceError

# 攻击类型 → 危险等级（按类别名索引）
_ATTACK_DANGER_MAP: Dict[str, str] = {
    'Normal': '无',
    'Analysis': '中危',
    'Backdoor': '严重',
    'DoS': '严重',
    'Exploits': '严重',
    'Fuzzers': '高危',
    'Generic': '高危',
    'Reconnaissance': '中危',
    'Shellcode': '严重',
    'Worms': '严重',
}


class TFLiteDetector(LoggerMixin):
    """
    TFLite 推理器

    与 IDSDetector 保持相同的 predict() 签名，
    支持 XNNPACK delegate 加速推理，自动处理 INT8 量化模型的输入/输出。
    """

    def __init__(
        self,
        model_path: str,
        input_dim: int = 48,
        num_classes: int = 10,
        sequence_length: int = 10,
        confidence_threshold: float = 0.5,
        alert_threshold: float = 0.8,
        scaler_path: Optional[str] = None,
        le_attack_path: Optional[str] = None,
        num_threads: int = 4,
        use_xnnpack: bool = True,
    ):
        """
        初始化 TFLite 推理器

        Args:
            model_path: .tflite 模型文件路径
            input_dim: 输入特征维度
            num_classes: 分类类别数
            sequence_length: 序列缓冲区长度
            confidence_threshold: 置信度阈值
            alert_threshold: 告警阈值
            scaler_path: Scaler 文件路径
            le_attack_path: LabelEncoder / 类别列表路径
            num_threads: 推理线程数（树莓派5 Cortex-A76 四核 = 4）
            use_xnnpack: 是否启用 XNNPACK delegate
        """
        super().__init__()

        if not HAS_TFLITE:
            raise ModelError(
                'TFLite 运行时未安装。'
                '树莓派: pip install tflite-runtime>=2.14.0; '
                'PC: pip install tensorflow>=2.14.0'
            )

        self.model_path = Path(model_path)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.alert_threshold = alert_threshold
        self.num_threads = num_threads
        self.use_xnnpack = use_xnnpack

        if not self.model_path.exists():
            raise ModelError(f'TFLite 模型文件不存在: {self.model_path}')

        # 加载 Scaler（训练时保存的标准化器）
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            import joblib
            self.scaler = joblib.load(scaler_path)
            if hasattr(self.scaler, 'feature_names_in_'):
                self.scaler.feature_names_in_ = None
            self.logger.info(f'已加载 Scaler: {scaler_path}')
        elif scaler_path:
            self.logger.warning(f'Scaler 文件不存在: {scaler_path}，将使用 Z-score 降级方案')

        # 加载 attack_cat 类别映射
        self._le_attack = None
        self._attack_type_names: Dict[int, str] = {}
        self._attack_danger_levels: Dict[int, str] = {}
        self._normal_class_id = 0

        if le_attack_path and Path(le_attack_path).exists():
            import joblib
            cat_order = joblib.load(le_attack_path)
            if isinstance(cat_order, list):
                classes = cat_order
            elif hasattr(cat_order, 'classes_'):
                classes = list(cat_order.classes_)
            else:
                classes = []
            self._le_attack = classes
            for i, name in enumerate(classes):
                self._attack_type_names[i] = name
                self._attack_danger_levels[i] = _ATTACK_DANGER_MAP.get(name, '未知')
                if name.lower() == 'normal':
                    self._normal_class_id = i
            self.logger.info(
                f'已加载 attack_cat 映射: {len(classes)} 类, Normal ID={self._normal_class_id}'
            )
        else:
            # 降级：使用内置默认映射
            _default_order = [
                'Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits',
                'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms',
            ]
            for i, name in enumerate(_default_order):
                self._attack_type_names[i] = name
                self._attack_danger_levels[i] = _ATTACK_DANGER_MAP.get(name, '未知')
            self._normal_class_id = 0
            if le_attack_path:
                self.logger.warning(f'LabelEncoder 文件不存在: {le_attack_path}，使用默认映射')

        # 加载 TFLite 模型
        self._load_model()

        # 序列缓冲区
        self._feature_buffer: deque = deque(maxlen=sequence_length)

        # 统计
        self._inference_times: deque = deque(maxlen=100)
        self._detection_count = 0
        self._attack_count = 0

        self.logger.info(
            f'TFLite 推理器就绪 | 模型: {self.model_path} | '
            f'输入维度: {input_dim} | 类别数: {num_classes} | '
            f'线程数: {num_threads} | XNNPACK: {use_xnnpack} | '
            f'后端: {TFLITE_BACKEND}'
        )

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self):
        """加载 TFLite 模型并配置 XNNPACK delegate"""
        # 配置 delegate
        delegates = []
        if self.use_xnnpack:
            try:
                # tflite-runtime 的 XNNPACK delegate
                xnnpack_delegate = tflite_interpreter.load_delegate(
                    library='libXNNPACK_delegate.so'
                    if TFLITE_BACKEND == 'tflite-runtime'
                    else None,
                )
                delegates.append(xnnpack_delegate)
                self.logger.info('XNNPACK delegate 已加载')
            except Exception as e:
                self.logger.warning(f'XNNPACK delegate 加载失败: {e}，使用默认 delegate')

        # 创建解释器
        if delegates:
            self.interpreter = tflite_interpreter.Interpreter(
                model_path=str(self.model_path),
                num_threads=self.num_threads,
                experimental_delegates=delegates,
            )
        else:
            self.interpreter = tflite_interpreter.Interpreter(
                model_path=str(self.model_path),
                num_threads=self.num_threads,
            )

        self.interpreter.allocate_tensors()

        # 获取输入/输出张量信息
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

        self._input_index = self._input_details[0]['index']
        self._output_index = self._output_details[0]['index']
        self._input_dtype = self._input_details[0]['dtype']
        self._output_dtype = self._output_details[0]['dtype']

        # 检查是否为 INT8 量化模型
        self._is_quantized = (self._input_dtype == np.int8 or self._input_dtype == np.uint8)
        if self._is_quantized:
            self._input_scale, self._input_zero_point = (
                self._input_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0],
                self._input_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0],
            )
            self._output_scale, self._output_zero_point = (
                self._output_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0],
                self._output_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0],
            )
            self.logger.info(
                f'INT8 量化模型 | 输入 scale={self._input_scale}, zp={self._input_zero_point} | '
                f'输出 scale={self._output_scale}, zp={self._output_zero_point}'
            )
        else:
            self._input_scale = 1.0
            self._input_zero_point = 0
            self._output_scale = 1.0
            self._output_zero_point = 0
            self.logger.info(f'FP32 模型 | 输入类型: {self._input_dtype}')

        # 验证输入形状
        input_shape = self._input_details[0]['shape']
        expected_shape = [1, self.sequence_length, self.input_dim]
        if input_shape != expected_shape:
            self.logger.warning(
                f'模型输入形状 {input_shape} 与期望 {expected_shape} 不一致，'
                f'将使用实际形状 {input_shape}'
            )

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    def _preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        预处理：标准化 + 序列缓冲 + 量化（如需要）

        Args:
            features: 原始特征向量 (input_dim,)

        Returns:
            预处理后的序列张量 (1, sequence_length, input_dim)
        """
        if len(features) != self.input_dim:
            raise InferenceError(
                f'特征维度不匹配: 期望 {self.input_dim}, 实际 {len(features)}'
            )

        # 使用训练时保存的 Scaler 进行标准化
        if self.scaler is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='X does not have valid feature names'
                )
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
        else:
            # 降级方案：Z-score 归一化
            mean, std = features.mean(), features.std()
            features = (features - mean) / (std + 1e-8)

        # 序列缓冲
        self._feature_buffer.append(features)
        while len(self._feature_buffer) < self.sequence_length:
            self._feature_buffer.append(features)

        sequence = np.array(list(self._feature_buffer)[-self.sequence_length:])

        # 添加 batch 维度
        sequence = sequence.reshape(1, self.sequence_length, self.input_dim)

        # INT8 量化模型的输入量化
        if self._is_quantized:
            sequence = self._quantize_input(sequence)

        return sequence.astype(self._input_dtype)

    def _quantize_input(self, data: np.ndarray) -> np.ndarray:
        """将 FP32 输入量化为 INT8"""
        quantized = data / self._input_scale + self._input_zero_point
        quantized = np.clip(quantized, -128, 127)
        return quantized.astype(np.int8)

    def _dequantize_output(self, data: np.ndarray) -> np.ndarray:
        """将 INT8 输出反量化为 FP32"""
        return (data.astype(np.float32) - self._output_zero_point) * self._output_scale

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> DetectionResult:
        """
        单次推理预测

        Args:
            features: 原始特征向量 (input_dim,)

        Returns:
            DetectionResult 检测结果
        """
        start = time.time()
        try:
            # 预处理
            input_tensor = self._preprocess(features)

            # 设置输入并推理
            self.interpreter.set_tensor(self._input_index, input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self._output_index)

            # 反量化（INT8 模型）
            if self._is_quantized:
                output = self._dequantize_output(output)

            probs = output[0]  # shape: (num_classes,)

            # 多分类: 取 softmax 最大值作为预测类别
            # 注意：TFLite 模型输出可能已经是 softmax 或 logits
            # 如果是 logits，需要 softmax；如果已是概率，直接取 argmax
            probs = self._ensure_probabilities(probs)

            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            # 攻击概率 = 所有非 Normal 类的概率和
            mask = np.ones(len(probs), dtype=bool)
            mask[self._normal_class_id] = False
            attack_prob = float(np.sum(probs[mask])) if len(probs) > 1 else 0.0

            # 攻击类型信息
            attack_type_id = pred_class
            attack_type_name = self._attack_type_names.get(
                pred_class, f'Unknown-{pred_class}'
            )
            danger_level = self._attack_danger_levels.get(pred_class, '未知')

            latency_ms = (time.time() - start) * 1000
            self._inference_times.append(latency_ms)
            self._detection_count += 1
            if pred_class != self._normal_class_id:
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
            self.logger.error(f'TFLite 推理错误: {e}')
            raise InferenceError(f'TFLite 推理失败: {e}')

    def _ensure_probabilities(self, raw_output: np.ndarray) -> np.ndarray:
        """
        确保输出为概率分布（softmax）
        如果输出值有负数或和不为 1，认为需要 softmax
        """
        # 如果所有值在 [0, 1] 且和接近 1，认为是概率
        if np.all(raw_output >= 0) and np.all(raw_output <= 1):
            total = np.sum(raw_output)
            if 0.9 < total < 1.1:
                return raw_output
        # 否则应用 softmax
        exp_x = np.exp(raw_output - np.max(raw_output))  # 数值稳定
        return exp_x / np.sum(exp_x)

    def predict_batch(self, features_list: List[np.ndarray]) -> List[DetectionResult]:
        """批量预测（逐个处理）"""
        return [self.predict(f) for f in features_list]

    def reset_buffer(self):
        """重置序列缓冲区"""
        self._feature_buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        times = list(self._inference_times)
        if not times:
            return {
                'total_detections': 0,
                'attack_count': 0,
                'backend': TFLITE_BACKEND,
            }
        return {
            'total_detections': self._detection_count,
            'attack_count': self._attack_count,
            'avg_latency_ms': round(np.mean(times), 2),
            'max_latency_ms': round(np.max(times), 2),
            'p95_latency_ms': (
                round(np.percentile(times, 95), 2)
                if len(times) >= 20
                else round(np.max(times), 2)
            ),
            'backend': TFLITE_BACKEND,
            'quantized': self._is_quantized,
            'num_threads': self.num_threads,
        }
