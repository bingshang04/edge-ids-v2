"""
ONNX → TensorFlow SavedModel → TFLite 转换脚本

完整链路:
1. ONNX → TensorFlow SavedModel (通过 onnx2tf 或 tf2onnx 逆向)
2. TensorFlow SavedModel → TFLite (FP32)
3. TFLite → INT8 量化（使用校准数据）

技术路线:
- 方案 A: onnx → onnx2tf → saved_model → tflite (推荐，支持 INT8 校准)
- 方案 B: onnx → tf2onnx(逆向) → saved_model → tflite
- 方案 C: onnx → onnxruntime 量化 → 不适用于 TFLite 部署

树莓派5 约束:
- 输入 shape: (1, 10, 48) [batch, seq_len, feat_dim]
- 输出 shape: (1, 10) [batch, num_classes]
- INT8 量化：输入/输出保持 FP32，内部 INT8
- 模型体积 < 5MB
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def onnx_to_tflite_via_onnx2tf(
    onnx_path: str,
    output_path: str,
    calibration_path: str = None,
    input_dim: int = 48,
    num_classes: int = 10,
    seq_len: int = 10,
):
    """
    通过 onnx2tf 将 ONNX 模型转为 INT8 TFLite

    要求: pip install onnx2tf
    """
    try:
        import onnx2tf
    except ImportError:
        logger.error("请先安装 onnx2tf: pip install onnx2tf")
        return None

    logger.info("方案 A: ONNX → onnx2tf → TFLite (INT8)")

    # onnx2tf 命令行调用
    import subprocess

    output_dir = str(Path(output_path).parent / 'tflite_export')
    calibrate_cmd = ''
    if calibration_path and Path(calibration_path).exists():
        calibrate_cmd = f'--quantize_int8 --calibration_data_path {calibration_path}'

    cmd = (
        f'onnx2tf -i "{onnx_path}" '
        f'-o "{output_dir}" '
        f'-osd '
        f'{calibrate_cmd} '
        f'--non_verbose'
    )

    logger.info(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"onnx2tf 转换失败:\n{result.stderr}")
        return None

    logger.info(f"onnx2tf 转换完成")

    # 复制生成的 TFLite 文件
    tflite_files = list(Path(output_dir).rglob('*.tflite'))
    if tflite_files:
        import shutil
        tflite_file = tflite_files[0]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tflite_file, output_path)
        file_size = os.path.getsize(output_path) / 1024
        logger.info(f"TFLite 模型已保存 → {output_path} ({file_size:.1f} KB)")
        if file_size > 5120:
            logger.warning(f"模型体积 {file_size:.1f} KB 超过 5MB 限制！")
        return output_path

    logger.error("未找到生成的 TFLite 文件")
    return None


def onnx_to_tflite_via_tensorflow(
    onnx_path: str,
    output_path: str,
    calibration_path: str = None,
    input_dim: int = 48,
    num_classes: int = 10,
    seq_len: int = 10,
):
    """
    通过 ONNX → TF SavedModel → TFLite 转换

    方案 B: 使用 tf2onnx 逆向 + TensorFlow Lite Converter
    """
    logger.info("方案 B: ONNX → TF SavedModel → TFLite")

    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        logger.error("请先安装 onnx-tf: pip install onnx-tf")
        return None

    # Step 1: 加载 ONNX
    logger.info(f"加载 ONNX 模型: {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    # Step 2: ONNX → TF
    logger.info("ONNX → TensorFlow...")
    tf_rep = prepare(onnx_model)

    # Step 3: TF → SavedModel
    saved_model_dir = str(Path(output_path).parent / 'saved_model')
    tf_rep.export_graph(saved_model_dir)
    logger.info(f"SavedModel 已保存 → {saved_model_dir}")

    # Step 4: SavedModel → TFLite
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("请先安装 tensorflow: pip install tensorflow")
        return None

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # INT8 量化（如果有校准数据）
    if calibration_path and Path(calibration_path).exists():
        logger.info(f"加载校准数据: {calibration_path}")
        calibration_data = np.load(calibration_path)
        logger.info(f"校准数据 shape: {calibration_data.shape}")

        def representative_dataset():
            for i in range(len(calibration_data)):
                # 取单条样本: (seq_len, input_dim) → (1, seq_len, input_dim)
                sample = calibration_data[i]
                if sample.ndim == 2:
                    sample = np.expand_dims(sample, axis=0)
                yield [sample.astype(np.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        # 输入/输出保持 FP32，内部 INT8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        # FP32 转换
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    logger.info("转换为 TFLite...")
    tflite_model = converter.convert()

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size = os.path.getsize(output_path) / 1024
    logger.info(f"TFLite 模型已保存 → {output_path} ({file_size:.1f} KB)")
    if file_size > 5120:
        logger.warning(f"模型体积 {file_size:.1f} KB 超过 5MB 限制！")

    # 验证 TFLite 模型
    _verify_tflite(output_path, input_dim, num_classes, seq_len)

    return output_path


def _verify_tflite(tflite_path, input_dim, num_classes, seq_len):
    """验证 TFLite 模型的输入输出"""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logger.info("TFLite 模型信息:")
        for inp in input_details:
            logger.info(f"  输入: name={inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")
        for out in output_details:
            logger.info(f"  输出: name={out['name']}, shape={out['shape']}, dtype={out['dtype']}")

        # 推理测试
        test_input = np.random.randn(1, seq_len, input_dim).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        logger.info(f"TFLite 推理验证通过，输出 shape: {output.shape}")
    except ImportError:
        logger.info("tensorflow 未安装，跳过 TFLite 验证")
    except Exception as e:
        logger.warning(f"TFLite 验证失败: {e}")


def convert(
    onnx_path: str = None,
    output_path: str = None,
    calibration_path: str = None,
    method: str = 'auto',
    input_dim: int = 48,
    num_classes: int = 10,
    seq_len: int = 10,
):
    """
    统一转换入口

    Args:
        onnx_path: ONNX 模型路径
        output_path: 输出 TFLite 路径
        calibration_path: 校准数据路径 (.npy)
        method: 转换方法 ('onnx2tf', 'tensorflow', 'auto')
        input_dim: 输入特征维度
        num_classes: 输出类别数
        seq_len: 序列长度
    """
    model_dir = ROOT_DIR / 'data' / 'models'

    if onnx_path is None:
        onnx_path = str(model_dir / 'tcn_model.onnx')
    if output_path is None:
        output_path = str(model_dir / 'tcn_model.tflite')
    if calibration_path is None:
        calib_path = model_dir / 'calibration_data.npy'
        if calib_path.exists():
            calibration_path = str(calib_path)
        else:
            logger.warning("校准数据未找到，将使用 FP32 转换（不做 INT8 量化）")

    logger.info(f"ONNX 输入: {onnx_path}")
    logger.info(f"TFLite 输出: {output_path}")
    logger.info(f"校准数据: {calibration_path}")

    # 自动选择转换方法
    if method == 'auto':
        try:
            import onnx2tf
            method = 'onnx2tf'
        except ImportError:
            try:
                import onnx_tf
                method = 'tensorflow'
            except ImportError:
                logger.error("请安装 onnx2tf 或 onnx-tf + tensorflow 来完成转换")
                logger.error("  pip install onnx2tf    # 推荐")
                logger.error("  pip install onnx-tf tensorflow  # 备选")
                return None

    logger.info(f"转换方法: {method}")

    if method == 'onnx2tf':
        return onnx_to_tflite_via_onnx2tf(
            onnx_path, output_path, calibration_path, input_dim, num_classes, seq_len
        )
    elif method == 'tensorflow':
        return onnx_to_tflite_via_tensorflow(
            onnx_path, output_path, calibration_path, input_dim, num_classes, seq_len
        )
    else:
        logger.error(f"未知转换方法: {method}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX → TFLite 转换（支持 INT8 量化）')
    parser.add_argument('--onnx', type=str, default=None,
                        help='ONNX 模型路径 (默认: data/models/tcn_model.onnx)')
    parser.add_argument('--output', type=str, default=None,
                        help='TFLite 输出路径 (默认: data/models/tcn_model.tflite)')
    parser.add_argument('--calibration', type=str, default=None,
                        help='校准数据路径 .npy (默认: data/models/calibration_data.npy)')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'onnx2tf', 'tensorflow'],
                        help='转换方法 (默认: auto)')
    parser.add_argument('--input_dim', type=int, default=48, help='输入特征维度')
    parser.add_argument('--num_classes', type=int, default=10, help='输出类别数')
    parser.add_argument('--seq_len', type=int, default=10, help='序列长度')
    args = parser.parse_args()

    convert(
        onnx_path=args.onnx,
        output_path=args.output,
        calibration_path=args.calibration,
        method=args.method,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
    )
