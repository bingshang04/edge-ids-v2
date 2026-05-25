"""
PyTorch 模型 → ONNX 导出脚本

将训练好的 ECA-TCN PyTorch 模型导出为 ONNX 格式。

注意事项:
- Chomp1d 的切片操作 x[:, :, :-chomp_size] 在 ONNX opset >= 11 中支持
- 使用 opset_version=14 以确保兼容性
- 支持动态 batch size 和固定/动态序列长度
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import torch
import argparse
import logging

from src.models.tcn_model import TCN

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    input_dim: int = 48,
    num_classes: int = 10,
    num_channels: list = None,
    kernel_size: int = 5,
    dropout: float = 0.3,
    use_eca: bool = True,
    seq_len: int = 10,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    dynamic_seq: bool = False,
):
    """
    将 PyTorch 模型导出为 ONNX 格式

    Args:
        model_path: PyTorch 模型权重路径 (.pth)
        output_path: 输出 ONNX 文件路径
        input_dim: 输入特征维度
        num_classes: 输出类别数
        num_channels: TCN 各层通道数
        kernel_size: 卷积核大小
        dropout: Dropout 概率
        use_eca: 是否使用 ECA 注意力
        seq_len: 序列长度
        opset_version: ONNX opset 版本
        dynamic_batch: 是否使用动态 batch size
        dynamic_seq: 是否使用动态序列长度
    """
    if num_channels is None:
        num_channels = [128, 256, 256]

    logger.info(f"输入 shape: (batch, {seq_len}, {input_dim})")
    logger.info(f"输出 shape: (batch, {num_classes})")
    logger.info(f"模型参数: channels={num_channels}, kernel={kernel_size}, dropout={dropout}, eca={use_eca}")

    # 实例化模型
    device = 'cpu'
    model = TCN(
        input_dim=input_dim,
        num_classes=num_classes,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        use_eca=use_eca,
    ).to(device)
    model.eval()

    # 加载权重
    if model_path and Path(model_path).exists():
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.info(f"已加载模型权重: {model_path}")
        logger.info(f"模型参数量: {model.num_params:,} ({model.num_params * 4 / 1024:.1f} KB FP32)")
    else:
        logger.warning(f"模型权重未找到: {model_path}，将导出随机初始化的模型")

    # 导出 ONNX
    dummy_input = torch.randn(1, seq_len, input_dim, device=device)

    # 动态轴配置
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes['input'] = {0: 'batch'}
        dynamic_axes['output'] = {0: 'batch'}
    if dynamic_seq:
        if 'input' in dynamic_axes:
            dynamic_axes['input'][1] = 'seq_len'
        else:
            dynamic_axes['input'] = {1: 'seq_len'}

    logger.info(f"导出 ONNX (opset={opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # 验证
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX 模型验证通过 ✓")

    # 模型信息
    onnx_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    logger.info(f"ONNX 输入 shape: {onnx_shape}")

    file_size = os.path.getsize(output_path) / 1024
    logger.info(f"ONNX 文件大小: {file_size:.1f} KB")
    logger.info(f"ONNX 模型已保存 → {output_path}")

    # 可选: 用 onnxruntime 做快速验证
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        test_input = dummy_input.numpy()
        outputs = session.run(None, {'input': test_input})
        logger.info(f"ONNX Runtime 推理验证通过，输出 shape: {outputs[0].shape}")
    except ImportError:
        logger.info("onnxruntime 未安装，跳过推理验证")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ECA-TCN → ONNX 导出')
    parser.add_argument('--model_path', type=str, default=None,
                        help='PyTorch 模型路径 (默认: data/models/tcn_model_3.0.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='ONNX 输出路径 (默认: data/models/tcn_model.onnx)')
    parser.add_argument('--input_dim', type=int, default=48, help='输入特征维度')
    parser.add_argument('--num_classes', type=int, default=10, help='输出类别数')
    parser.add_argument('--seq_len', type=int, default=10, help='序列长度')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset 版本')
    parser.add_argument('--no_dynamic_batch', action='store_true', help='禁用动态 batch size')
    args = parser.parse_args()

    model_dir = ROOT_DIR / 'data' / 'models'
    model_path = args.model_path or str(model_dir / 'tcn_model_3.0.pth')
    output_path = args.output or str(model_dir / 'tcn_model.onnx')

    export_to_onnx(
        model_path=model_path,
        output_path=output_path,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        opset_version=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
    )
