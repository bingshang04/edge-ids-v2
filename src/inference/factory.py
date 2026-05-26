"""
推理器工厂模块
按平台自动选择推理后端，支持单模型和两步级联
"""
import logging
from typing import Optional, Union
from pathlib import Path

from ..utils.platform_info import get_platform_type

logger = logging.getLogger(__name__)


def _create_single_detector(config, model_path_override=None, num_classes_override=None,
                            num_channels_override=None, backend='pytorch'):
    """创建单个推理器（PyTorch 或 TFLite）"""
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / 'data' / 'models'

    model_path = model_path_override or config.model.model_path
    num_classes = num_classes_override or getattr(config.model, 'num_classes', 9)

    if backend == 'tflite':
        from .tflite_detector import TFLiteDetector
        tflite_path = model_path
        if tflite_path and not Path(tflite_path).is_absolute():
            tflite_path = str(project_root / tflite_path)
        num_threads = getattr(config.inference, 'tflite_num_threads', 4)
        return TFLiteDetector(
            model_path=tflite_path or str(model_dir / 'tcn_model.tflite'),
            input_dim=config.model.input_dim,
            num_classes=num_classes,
            sequence_length=config.inference.sequence_length,
            confidence_threshold=config.inference.confidence_threshold,
            alert_threshold=config.inference.alert_threshold,
            scaler_path=config.inference.scaler_path,
            le_attack_path=str(model_dir / 'le_attack_cat.joblib'),
            num_threads=num_threads,
            use_xnnpack=True,
        )
    else:
        from .detector import IDSDetector
        channels = num_channels_override or (
            config.model.num_channels if hasattr(config, 'model') else [128, 256, 256])
        return IDSDetector(
            model_path=model_path,
            input_dim=config.model.input_dim,
            num_classes=num_classes,
            num_channels=channels,
            kernel_size=getattr(config.model, 'kernel_size', 5),
            dropout=getattr(config.model, 'dropout', 0.3),
            use_eca=getattr(config.model, 'use_eca', True),
            sequence_length=config.inference.sequence_length,
            confidence_threshold=config.inference.confidence_threshold,
            alert_threshold=config.inference.alert_threshold,
            scaler_path=config.inference.scaler_path,
            le_attack_path=str(model_dir / 'le_attack_cat.joblib'),
        )


def create_detector(config=None):
    """
    按平台自动选择推理后端（支持两步级联）

    决策逻辑：
      - config.model.model_b_path 存在 → TwoStageDetector
      - 树莓派5 + use_tflite=True → TFLiteDetector
      - 其他 → IDSDetector (PyTorch)
    """
    if config is None:
        from .detector import IDSDetector
        logger.info('无配置，使用默认 PyTorch IDSDetector')
        return IDSDetector()

    platform_type = get_platform_type()
    use_tflite = getattr(config.inference, 'use_tflite', False)

    # 判断是否两步级联模式
    has_model_b = (hasattr(config.model, 'model_b_path') and config.model.model_b_path)

    if has_model_b:
        from .two_stage_detector import TwoStageDetector
        logger.info(f'两步级联模式: Model_A(二分类) + Model_B(九分类)')

        if platform_type == 'raspberry_pi' and use_tflite:
            # TFLite 级联
            tflite_a = getattr(config.inference, 'tflite_model_a_path', None)
            tflite_b = getattr(config.inference, 'tflite_model_b_path', None)
            model_a = _create_single_detector(
                config, model_path_override=tflite_a,
                num_classes_override=2, backend='tflite')
            model_b = _create_single_detector(
                config, model_path_override=tflite_b,
                num_classes_override=9, backend='tflite')
        else:
            # PyTorch 级联
            model_a = _create_single_detector(
                config, model_path_override=config.model.model_a_path,
                num_classes_override=2,
                num_channels_override=config.model.model_a_channels)
            model_b = _create_single_detector(
                config, model_path_override=config.model.model_b_path,
                num_classes_override=9,
                num_channels_override=config.model.model_b_channels)

        return TwoStageDetector(model_a, model_b)

    # 单模型模式（向后兼容）
    if platform_type == 'raspberry_pi' and use_tflite:
        logger.info(f'平台 [{platform_type}] → TFLiteDetector (XNNPACK)')
        return _create_single_detector(config, backend='tflite')
    else:
        logger.info(f'平台 [{platform_type}] → PyTorch IDSDetector')
        return _create_single_detector(config, backend='pytorch')
