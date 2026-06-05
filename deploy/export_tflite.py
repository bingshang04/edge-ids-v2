"""
模型导出脚本 — PyTorch → ONNX → TensorFlow Lite

导出流程:
  1. PyTorch (.pt) → ONNX (.onnx)
  2. ONNX → TensorFlow (SavedModel)
  3. TensorFlow → TFLite (FP32 / INT8量化)

目标: 生成可在树莓派5上通过 TFLite Runtime 高效推理的模型
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ds_tcn_ids import DSTCNIDS, create_model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 100, 49),
    opset_version: int = 14,
):
    """
    PyTorch → ONNX

    Args:
        model: DS-TCN-IDS 模型
        output_path: .onnx 文件路径
        input_shape: (batch, seq_len, features)
        opset_version: ONNX opset版本
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"✅ ONNX 模型已导出: {output_path}")

    # 验证 ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型验证通过")
    except ImportError:
        print("⚠️ onnx 未安装，跳过验证 (pip install onnx)")

    return output_path


def onnx_to_tflite(
    onnx_path: str,
    output_path: str,
    quantization: str = "fp32",
    representative_dataset: np.ndarray | None = None,
):
    """
    ONNX → TensorFlow Lite

    Args:
        onnx_path: .onnx 模型路径
        output_path: .tflite 输出路径
        quantization: 'fp32' | 'fp16' | 'int8'
        representative_dataset: INT8量化的校准数据集 (约500样本)
    """
    # ONNX → TF SavedModel
    try:
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_path = output_path.replace(".tflite", "_tf")
        tf_rep.export_graph(tf_path)
        print(f"✅ TensorFlow SavedModel: {tf_path}")

    except ImportError:
        print("⚠️ onnx-tf 未安装，尝试直接 ONNX → TFLite")
        # 备选: 直接转 (需要 TensorFlow)
        pass

    # TF → TFLite
    try:
        import tensorflow as tf

        # 加载 SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

        if quantization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]

            if representative_dataset is not None:
                def representative_data_gen():
                    for i in range(min(len(representative_dataset), 500)):
                        sample = representative_dataset[i:i+1]
                        yield [sample.astype(np.float32)]

                converter.representative_dataset = representative_data_gen
            else:
                print("⚠️ INT8量化需要校准数据集，降级为动态范围量化")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            # FP32 不量化
            pass

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        # 检查模型大小
        size_kb = len(tflite_model) / 1024
        print(f"✅ TFLite 模型已导出: {output_path} ({size_kb:.1f} KB)")

    except ImportError:
        print("⚠️ TensorFlow 未安装，跳过 TFLite 转换")
        print("  在树莓派5上运行: pip install tensorflow")

    return output_path


def export_model(
    model: torch.nn.Module,
    output_dir: str = "./deploy/exported/",
    model_name: str = "ds-tcn-ids",
    input_shape: tuple = (1, 100, 49),
    quantization: str = "fp32",
    calibration_data: np.ndarray | None = None,
):
    """
    一键导出: PyTorch → ONNX → TFLite

    Args:
        model: DS-TCN-IDS 模型
        output_dir: 输出目录
        model_name: 模型名称
        input_shape: 输入形状
        quantization: 量化方案
        calibration_data: INT8校准数据
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: PyTorch → ONNX
    onnx_path = str(output_path / f"{model_name}.onnx")
    export_to_onnx(model, onnx_path, input_shape)

    # Step 2: ONNX → TFLite
    tflite_path = str(output_path / f"{model_name}_{quantization}.tflite")

    try:
        onnx_to_tflite(onnx_path, tflite_path, quantization, calibration_data)
    except Exception as e:
        print(f"⚠️ TFLite 转换失败: {e}")
        print("  备选方案: 使用 onnxruntime 在树莓派5上直接运行 ONNX")

    # 同时保存 PyTorch 权重
    torch_path = str(output_path / f"{model_name}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "num_classes": model.num_classes,
        },
        torch_path,
    )
    print(f"✅ PyTorch 权重已保存: {torch_path}")

    return str(output_path)


def benchmark_tflite(
    tflite_path: str,
    test_data: np.ndarray,
    n_warmup: int = 10,
    n_runs: int = 100,
):
    """
    TFLite 推理性能基准测试

    Args:
        tflite_path: .tflite 模型路径
        test_data: 测试数据 (N, seq_len, features)
        n_warmup: 预热次数
        n_runs: 测试运行次数
    """
    try:
        import tensorflow as tf
        import time

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 预热
        for i in range(n_warmup):
            interpreter.set_tensor(input_details[0]["index"], test_data[i:i+1])
            interpreter.invoke()

        # 计时
        latencies = []
        for i in range(n_runs):
            sample = test_data[i % len(test_data):(i % len(test_data)) + 1]
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], sample)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]["index"])
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)
        print(f"\n📊 TFLite 推理性能 (Python, CPU):")
        print(f"   平均延迟: {latencies.mean():.2f} ms")
        print(f"   P50: {np.percentile(latencies, 50):.2f} ms")
        print(f"   P95: {np.percentile(latencies, 95):.2f} ms")
        print(f"   P99: {np.percentile(latencies, 99):.2f} ms")
        print(f"   吞吐量: {1000/latencies.mean():.1f} samples/s")

        return latencies

    except ImportError:
        print("⚠️ TensorFlow 未安装，无法运行 TFLite 基准测试")
        return None


if __name__ == "__main__":
    # 演示导出流程
    print("DS-TCN-IDS 模型导出\n")

    # 创建 tiny 模型 (树莓派5部署)
    model = create_model(input_dim=49, num_classes=10, model_size="tiny")
    params, size_mb = model.get_model_size()
    print(f"模型: tiny, {params:,} 参数, {size_mb:.2f}MB (FP32)")

    # 导出 ONNX
    onnx_path = export_to_onnx(
        model,
        "./deploy/exported/ds-tcn-ids-tiny.onnx",
        input_shape=(1, 100, 49),
    )

    # 基准测试 (如果没有 TensorFlow，至少验证 ONNX)
    dummy_data = np.random.randn(50, 100, 49).astype(np.float32)

    # PyTorch 推理基准
    import time
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(5):
            _ = model(torch.FloatTensor(dummy_data[:1]))
        # 计时
        latencies = []
        for i in range(100):
            sample = torch.FloatTensor(dummy_data[i % 50:(i % 50) + 1])
            start = time.perf_counter()
            _ = model(sample)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        print(f"\n📊 PyTorch 推理性能 (CPU):")
        print(f"   平均延迟: {latencies.mean():.2f} ms")
        print(f"   P95: {np.percentile(latencies, 95):.2f} ms")

    # 尝试 TFLite 导出
    try:
        export_model(
            model,
            output_dir="./deploy/exported/",
            model_name="ds-tcn-ids-tiny",
            input_shape=(1, 100, 49),
            quantization="fp32",
        )
    except Exception as e:
        print(f"TFLite 导出需要 TensorFlow 环境: {e}")
