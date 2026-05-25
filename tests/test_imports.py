"""
Edge-IDS v2.0 模块导入测试
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))


def test_imports():
    errors = []

    # 配置模块
    try:
        from src.config import get_settings
        from src.config.constants import PROJECT_NAME, PROJECT_VERSION
        print(f"[OK] 配置模块 — {PROJECT_NAME} v{PROJECT_VERSION}")
    except Exception as e:
        errors.append(f"配置模块: {e}")
        print(f"[FAIL] 配置模块: {e}")

    # 工具模块
    try:
        from src.utils.logger import get_logger
        from src.utils.exceptions import EdgeIDSException
        from src.utils.helpers import ensure_dir, get_flow_id
        from src.utils.platform_info import get_platform_type
        print(f"[OK] 工具模块 — 平台: {get_platform_type()}")
    except Exception as e:
        errors.append(f"工具模块: {e}")
        print(f"[FAIL] 工具模块: {e}")

    # ECA 注意力模块
    try:
        from src.models.eca import ECALayer
        import torch
        eca = ECALayer(128)
        x = torch.randn(2, 128, 20)
        y = eca(x)
        assert y.shape == x.shape, f"ECA 输出shape错误: {y.shape}"
        print(f"[OK] ECA 注意力模块 — kernel_size={eca.kernel_size}")
    except Exception as e:
        errors.append(f"ECA模块: {e}")
        print(f"[FAIL] ECA模块: {e}")

    # 模型模块
    try:
        from src.models.tcn_model import TCN
        model = TCN(input_dim=48, num_classes=2, num_channels=[128, 256, 256], use_eca=True)
        info = model.get_model_info()
        print(f"[OK] TCN 模型 — {info['num_parameters'] / 1e6:.2f}M 参数, ECA={info['use_eca']}")
    except Exception as e:
        errors.append(f"模型模块: {e}")
        print(f"[FAIL] 模型模块: {e}")

    # 捕获模块
    try:
        from src.capture.packet_capture import PacketCapture, PacketInfo
        print(f"[OK] 捕获模块")
    except Exception as e:
        errors.append(f"捕获模块: {e}")
        print(f"[FAIL] 捕获模块: {e}")

    # 特征模块
    try:
        from src.features.flow_features import FeatureExtractor, FEATURE_DIM
        print(f"[OK] 特征模块 — 特征维度: {FEATURE_DIM}")
    except Exception as e:
        errors.append(f"特征模块: {e}")
        print(f"[FAIL] 特征模块: {e}")

    # 推理模块
    try:
        from src.inference.detector import IDSDetector, DetectionResult
        print(f"[OK] 推理模块")
    except Exception as e:
        errors.append(f"推理模块: {e}")
        print(f"[FAIL] 推理模块: {e}")

    # 主程序
    try:
        from main import EdgeIDS
        print(f"[OK] 主程序")
    except Exception as e:
        errors.append(f"主程序: {e}")
        print(f"[FAIL] 主程序: {e}")

    print("\n" + "=" * 50)
    if not errors:
        print(f"[OK] 所有 {7} 个模块导入成功！")
        return True
    else:
        print(f"[FAIL] {len(errors)} 个模块导入失败:")
        for e in errors:
            print(f"  - {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
