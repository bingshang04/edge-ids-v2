"""
推理器工厂模块
按平台自动选择推理后端（TFLite / PyTorch）
"""
import logging
from typing import Optional, Union

from ..utils.platform_info import get_platform_type

logger = logging.getLogger(__name__)


def create_detector(config=None):
    """
    按平台自动选择推理后端

    决策逻辑：
      - 树莓派5 + config.inference.use_tflite=True → TFLiteDetector（XNNPACK）
      - 其他平台 → IDSDetector（PyTorch）

    Args:
        config: Settings 实例

    Returns:
        TFLiteDetector | IDSDetector 实例
    """
    if config is None:
        # 无配置时降级为 PyTorch IDSDetector
        from .detector import IDSDetector
        logger.info('无配置，使用默认 PyTorch IDSDetector')
        return IDSDetector()

    platform_type = get_platform_type()
    use_tflite = getattr(config.inference, 'use_tflite', False)

    if platform_type == 'raspberry_pi' and use_tflite:
        # 树莓派5 + TFLite 推理
        from .tflite_detector import TFLiteDetector
        from pathlib import Path

        tflite_path = getattr(config.inference, 'tflite_model_path', None)
        num_threads = getattr(config.inference, 'tflite_num_threads', 4)
        project_root = Path(__file__).parent.parent.parent

        if tflite_path and not Path(tflite_path).is_absolute():
            tflite_path = str(project_root / tflite_path)

        model_dir = project_root / 'data' / 'models'

        logger.info(f'平台 [{platform_type}] → 使用 TFLite 推理器 (XNNPACK)')
        return TFLiteDetector(
            model_path=tflite_path or str(model_dir / 'tcn_model.tflite'),
            input_dim=config.model.input_dim,
            num_classes=getattr(config.model, 'num_classes', 10),
            sequence_length=config.inference.sequence_length,
            confidence_threshold=config.inference.confidence_threshold,
            alert_threshold=config.inference.alert_threshold,
            scaler_path=config.inference.scaler_path,
            le_attack_path=str(model_dir / 'le_attack_cat.joblib'),
            num_threads=num_threads,
            use_xnnpack=True,
        )

    else:
        # 其他平台 → PyTorch IDSDetector
        from .detector import IDSDetector, create_detector as _create_pt_detector
        logger.info(f'平台 [{platform_type}] → 使用 PyTorch IDSDetector')
        return _create_pt_detector(config)
