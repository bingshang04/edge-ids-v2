"""
校准数据集提取脚本 — 从 UNSW-NB15 验证集提取 500-1000 条特征向量
用于 TFLite INT8 量化的校准数据
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def extract_calibration_data(num_samples=800, output_path=None, random_seed=42):
    """
    从 UNSW-NB15 测试集提取校准样本

    Args:
        num_samples: 提取的样本数 (500-1000)
        output_path: 输出 .npy 文件路径
        random_seed: 随机种子

    Returns:
        calibration_data: (num_samples, 10, 48) 的 numpy 数组
    """
    DATA_DIR = ROOT_DIR / 'data' / 'raw'
    TEST_PATH = DATA_DIR / 'UNSW_NB15_testing-set.csv'

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {TEST_PATH}")

    logger.info(f"加载测试数据: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    logger.info(f"测试集: {test_df.shape}")

    # 特征工程（与 train.py 一致）
    numeric_cols = [col for col in test_df.columns
                    if col not in ['id', 'proto', 'service', 'state', 'attack_cat', 'label']]

    test_df['byte_ratio'] = test_df['sbytes'] / (test_df['dbytes'] + 1e-6)
    test_df['load_ratio'] = test_df['sload'] / (test_df['dload'] + 1e-6)
    test_df['pkt_ratio'] = test_df['spkts'] / (test_df['spkts'] + test_df['dpkts'] + 1e-6)
    test_df['dur_rate'] = test_df['dur'] * test_df['rate']
    test_df['ttl_diff'] = test_df['sttl'] - test_df['dttl']
    test_df['avg_pkt_size'] = (test_df['sbytes'] + test_df['dbytes']) / (test_df['spkts'] + test_df['dpkts'] + 1e-6)

    # 类别编码（用简单映射，只需要数值即可）
    from sklearn.preprocessing import LabelEncoder
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        le = LabelEncoder()
        test_df[col] = le.fit_transform(test_df[col].fillna('unknown').astype(str))

    feature_cols = numeric_cols + ['byte_ratio', 'load_ratio', 'pkt_ratio',
                                   'dur_rate', 'ttl_diff', 'avg_pkt_size'] + cat_cols
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # 标准化（使用简单 StandardScaler）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(test_df[feature_cols]).astype(np.float32)

    # 采样 + 创建时序序列 (seq_len=10)
    np.random.seed(random_seed)
    seq_length = 10

    # 创建时序窗口
    sequences = []
    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i:i + seq_length])

    sequences = np.array(sequences, dtype=np.float32)
    logger.info(f"时序序列总数: {sequences.shape}")

    # 随机选取 num_samples 条
    if len(sequences) > num_samples:
        indices = np.random.choice(len(sequences), num_samples, replace=False)
        calibration_data = sequences[indices]
    else:
        calibration_data = sequences

    logger.info(f"校准数据集: {calibration_data.shape}")

    # 保存
    if output_path is None:
        output_path = ROOT_DIR / 'data' / 'models' / 'calibration_data.npy'
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, calibration_data)
    logger.info(f"校准数据已保存 → {output_path}")

    return calibration_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNSW-NB15 校准数据集提取')
    parser.add_argument('--num_samples', type=int, default=800,
                        help='提取的校准样本数 (默认: 800)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径 (默认: data/models/calibration_data.npy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    args = parser.parse_args()

    extract_calibration_data(
        num_samples=args.num_samples,
        output_path=args.output,
        random_seed=args.seed,
    )
