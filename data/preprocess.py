"""
UNSW-NB15 数据预处理管道

数据来源:
  1. Kaggle: kagglehub.dataset_download('mrwellsdavid/unsw-nb15')
  2. 手动下载 CSV 放入 data/raw/UNSW-NB15/

处理流程:
  CSV → 清洗 → 编码 → 标准化 → 序列构建 → .npy

用法:
  python data/preprocess.py                          # 自动查找数据
  python data/preprocess.py --data-dir ./data/raw/UNSW-NB15/
  python data/preprocess.py --window 100 --stride 1  # 序列参数
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# UNSW-NB15 攻击类别映射
ATTACK_CATEGORIES = {
    "Normal": 0,
    "Fuzzers": 1,
    "Analysis": 2,
    "Backdoor": 3,
    "DoS": 4,
    "Exploits": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9,
}
NUM_CLASSES = len(ATTACK_CATEGORIES)

# 需要编码的类别特征
CATEGORICAL_COLS = ["proto", "service", "state"]

# 需要删除的非特征列
DROP_COLS = ["id", "srcip", "dstip", "attack_cat", "label"]


def find_data(data_dir: str) -> tuple[str | None, str | None]:
    """自动查找 UNSW-NB15 CSV 文件"""
    data_path = Path(data_dir)

    train_file = None
    test_file = None

    # 搜索模式
    train_patterns = ["UNSW_NB15_training-set.csv", "UNSW_NB15_training-set.csv"]
    test_patterns = ["UNSW_NB15_testing-set.csv", "UNSW_NB15_testing-set.csv"]

    for f in data_path.rglob("*.csv"):
        fname = f.name
        if "training" in fname.lower():
            train_file = str(f)
        elif "testing" in fname.lower():
            test_file = str(f)

    # 尝试 Kaggle 缓存
    if train_file is None:
        kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "mrwellsdavid" / "unsw-nb15"
        if kaggle_cache.exists():
            for vdir in kaggle_cache.iterdir():
                for f in vdir.glob("*.csv"):
                    if "training" in f.name.lower():
                        train_file = str(f)
                    elif "testing" in f.name.lower():
                        test_file = str(f)

    return train_file, test_file


def load_data(train_path: str, test_path: str | None) -> pd.DataFrame:
    """加载并合并训练集和测试集"""
    print(f"📂 加载训练集: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"   训练集: {train_df.shape}")

    if test_path and os.path.exists(test_path):
        print(f"📂 加载测试集: {test_path}")
        test_df = pd.read_csv(test_path)
        print(f"   测试集: {test_df.shape}")
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    n_before = len(df)

    # 替换无穷大
    df = df.replace([np.inf, -np.inf], np.nan)

    # 删除 NaN 行 (论文2做法)
    df = df.dropna()
    n_after_nan = len(df)

    # 删除全零列
    zero_cols = [c for c in df.columns if (df[c] == 0).all() and c not in DROP_COLS]
    if zero_cols:
        print(f"  删除全零列: {zero_cols}")
        df = df.drop(columns=zero_cols)

    print(f"  清洗: {n_before} → {n_after_nan} (删除 {n_before - n_after_nan} 行)")
    return df


def extract_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """提取攻击类别标签"""
    if "attack_cat" in df.columns:
        labels = df["attack_cat"].map(ATTACK_CATEGORIES)
        labels = labels.fillna(0).astype(np.int64).values  # 未知→Normal
    elif "label" in df.columns:
        # 只有二分类标签的情况
        labels = df["label"].values.astype(np.int64)
    else:
        labels = np.zeros(len(df), dtype=np.int64)

    return df, labels


def encode_and_normalize(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, int]:
    """特征编码 + Z-Score标准化"""
    # 删除不需要的列
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Label Encoding 类别特征
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 只保留数值列
    numeric_df = df.select_dtypes(include=[np.number])
    num_features = numeric_df.shape[1]
    print(f"  特征维度: {num_features}")

    # Z-Score 标准化
    scaler = StandardScaler()
    data = scaler.fit_transform(numeric_df.values).astype(np.float32)

    return data, labels, scaler, num_features


def build_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 100,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口构建序列"""
    n = len(data)
    sequences, seq_labels = [], []

    # 确定步数
    indices = list(range(0, n - window_size + 1, stride))
    print(f"  构建序列: {len(indices)} 个窗口 (窗口={window_size}, 步长={stride})")

    for start in tqdm(indices, desc="  📐 序列构建", unit="seq"):
        end = start + window_size
        seq = data[start:end]
        seq_lbls = labels[start:end]

        # 窗口标签: 任一异常 → 该攻击类型
        attack_mask = seq_lbls > 0
        if np.any(attack_mask):
            seq_label = seq_lbls[attack_mask][0]
        else:
            seq_label = 0

        sequences.append(seq)
        seq_labels.append(seq_label)

    return np.array(sequences, dtype=np.float32), np.array(seq_labels, dtype=np.int64)


def generate_synthetic_data(
    n_samples: int = 100000,
    n_features: int = 42,
    window_size: int = 100,
    output_dir: str = "./data/processed/",
    seed: int = 42,
):
    """
    生成模拟 UNSW-NB15 特征的合成数据 (当无法获取真实数据时使用)

    模拟 UNSW-NB15 的数据特征:
      - 类别极度不平衡 (~87% 正常, ~13% 攻击)
      - 攻击类别呈长尾分布
      - 数值特征之间有一定的相关性
    """
    print("\n⚠️  未找到真实数据，生成模拟数据...")
    print(f"   样本: {n_samples}, 特征: {n_features}, 窗口: {window_size}")
    np.random.seed(seed)

    # 模拟 UNSW-NB15 类别分布 (基于论文2 表2)
    class_dist = {
        0: 0.4787,   # Normal (约48%)
        1: 0.0728,   # Fuzzers
        2: 0.0054,   # Analysis
        3: 0.0056,   # Backdoor
        4: 0.0390,   # DoS
        5: 0.0890,   # Exploits
        6: 0.1200,   # Generic
        7: 0.0716,   # Reconnaissance
        8: 0.0034,   # Shellcode
        9: 0.0028,   # Worms (极少)
    }

    # 为每个攻击类生成不同分布的特征
    # 正常流量: 特征值集中在均值附近
    # 攻击流量: 特征值有偏移
    normal_data = np.random.randn(
        int(n_samples * 0.7), n_features
    ).astype(np.float32)

    attack_data_list = [normal_data]
    attack_labels_list = [np.zeros(len(normal_data), dtype=np.int64)]

    for cls_id in range(1, NUM_CLASSES):
        n_cls = max(100, int(n_samples * class_dist[cls_id]))
        # 攻击类特征有偏移 (模拟真实攻击特征差异)
        shift = np.random.randn(n_features) * 2.0
        scale = np.random.uniform(0.5, 2.0, n_features)
        cls_data = (np.random.randn(n_cls, n_features) * scale + shift).astype(np.float32)
        attack_data_list.append(cls_data)
        attack_labels_list.append(np.full(n_cls, cls_id, dtype=np.int64))

    all_data = np.concatenate(attack_data_list)
    all_labels = np.concatenate(attack_labels_list)

    # 打乱
    perm = np.random.permutation(len(all_data))
    all_data = all_data[perm]
    all_labels = all_labels[perm]

    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data).astype(np.float32)

    print(f"   总样本: {len(all_data)}, 类别分布:")
    for cls_id in range(NUM_CLASSES):
        count = (all_labels == cls_id).sum()
        cls_name = list(ATTACK_CATEGORIES.keys())[cls_id]
        print(f"     {cls_name}: {count} ({count/len(all_labels)*100:.1f}%)")

    # 构建序列
    X, y = build_sequences(all_data, all_labels, window_size, stride=10)

    # 拆分
    n = len(X)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed,
    )

    save_processed(X_train, y_train, X_val, y_val, X_test, y_test, output_dir)

    return {
        "num_features": n_features,
        "num_classes": NUM_CLASSES,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "window_size": window_size,
        "is_synthetic": True,
    }


def save_processed(
    X_train, y_train, X_val, y_val, X_test, y_test, output_dir: str,
):
    """保存处理后的数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "y_val.npy", y_val)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_test.npy", y_test)

    print(f"\n✅ 数据已保存: {output_path}")
    print(f"   训练: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"   验证: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"   测试: X_test={X_test.shape}, y_test={y_test.shape}")


def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 数据预处理")
    parser.add_argument("--data-dir", default="./data/raw/UNSW-NB15/")
    parser.add_argument("--output-dir", default="./data/processed/")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Edge-IDS 数据预处理")
    print("=" * 60)

    # 查找数据
    train_path, test_path = find_data(args.data_dir)

    if train_path is None:
        # 生成合成数据
        info = generate_synthetic_data(
            n_samples=100000,
            n_features=42,
            window_size=args.window,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        # 加载真实数据
        df = load_data(train_path, test_path)
        df = clean_data(df)
        df, labels = extract_labels(df)

        print(f"\n  类别分布:")
        for cls_name, cls_id in sorted(ATTACK_CATEGORIES.items(), key=lambda x: x[1]):
            count = (labels == cls_id).sum()
            if count > 0:
                print(f"    {cls_name}: {count} ({count/len(labels)*100:.2f}%)")

        # 编码 + 标准化
        data, labels, scaler, n_features = encode_and_normalize(df, labels)

        # 构建序列
        X, y = build_sequences(data, labels, args.window, args.stride)

        # 拆分: 训练/验证/测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=args.test_size, stratify=y_temp, random_state=args.seed,
        )

        save_processed(X_train, y_train, X_val, y_val, X_test, y_test, args.output_dir)

        info = {
            "num_features": n_features,
            "num_classes": NUM_CLASSES,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "window_size": args.window,
            "is_synthetic": False,
        }

    print(f"\n📊 数据信息: {info}")
    print("✅ 预处理完成！运行训练：python train/train.py")


if __name__ == "__main__":
    main()
