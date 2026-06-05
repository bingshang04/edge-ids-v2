"""
UNSW-NB15 数据预处理管道 (v2 重构)

改进:
  - 48 维特征 (39数值 + 6衍生 + 3类别编码)
  - 保存原始记录（不在此处做滑动窗口，由训练脚本处理）
  - 合并特征工程 + 标准化 + 标签构建

数据流程:
  CSV → 清洗 → 衍生特征 → 类别编码 → StandardScaler → .npy + .joblib

用法:
  python data/preprocess.py
  python data/preprocess.py --data-dir ./data/raw/UNSW-NB15/
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


# ============================================================
# 配置常量
# ============================================================

# 10 类 (保留 DoS / Exploits 独立，不合并)
CATEGORY_ORDER = [
    "Normal", "Fuzzers", "Analysis", "Backdoor",
    "DoS", "Exploits", "Generic", "Reconnaissance",
    "Shellcode", "Worms",
]

ATTACK_CAT_MAP = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
NUM_CLASSES = len(CATEGORY_ORDER)

# 需要编码的类别特征
CAT_COLS = ["proto", "service", "state"]

# 需要映射到数值的列
NUMERIC_COLS = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]


def load_csv_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载训练集和测试集 CSV"""
    data_path = Path(data_dir)
    train_files = list(data_path.glob("*training*.csv"))
    test_files = list(data_path.glob("*testing*.csv"))

    if not train_files:
        # 尝试 Kaggle 缓存
        kaggle_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "mrwellsdavid" / "unsw-nb15"
        if kaggle_dir.exists():
            for vdir in kaggle_dir.iterdir():
                train_files.extend(vdir.glob("*training*.csv"))
                test_files.extend(vdir.glob("*testing*.csv"))

    if not train_files:
        raise FileNotFoundError(
            f"未找到 UNSW-NB15 CSV 文件。请下载后放入 {data_dir}\n"
            f"或使用 Kaggle 命令: kagglehub.dataset_download('mrwellsdavid/unsw-nb15')"
        )

    train_path = str(train_files[0])
    test_path = str(test_files[0]) if test_files else None

    print(f"📂 加载训练集: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"   训练集: {train_df.shape}")

    if test_path:
        print(f"📂 加载测试集: {test_path}")
        test_df = pd.read_csv(test_path)
        print(f"   测试集: {test_df.shape}")
    else:
        test_df = pd.DataFrame()

    return train_df, test_df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加 6 个衍生特征 (来自 edge_ids_v2 的经验)"""
    df = df.copy()

    # 流量比例
    df["byte_ratio"] = df["sbytes"] / (df["dbytes"] + 1e-6)
    df["load_ratio"] = df["sload"] / (df["dload"] + 1e-6)
    df["pkt_ratio"] = df["spkts"] / (df["spkts"] + df["dpkts"] + 1e-6)

    # 交互特征
    df["dur_rate"] = df["dur"] * df["rate"]
    df["ttl_diff"] = df["sttl"] - df["dttl"]

    # 包大小
    df["avg_pkt_size"] = (df["sbytes"] + df["dbytes"]) / (df["spkts"] + df["dpkts"] + 1e-6)

    return df


def encode_and_normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """类别编码 + Z-Score 标准化 + 标签构建"""

    # --- 衍生特征 ---
    train_df = add_derived_features(train_df)
    if len(test_df) > 0:
        test_df = add_derived_features(test_df)

    # --- 类别特征编码 ---
    label_encoders = {}
    for col in CAT_COLS:
        combined = pd.concat(
            [train_df[col], test_df[col]] if len(test_df) > 0 else [train_df[col]],
            axis=0,
        ).fillna("unknown").astype(str)
        le = LabelEncoder()
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].fillna("unknown").astype(str))
        if len(test_df) > 0:
            test_df[col] = le.transform(test_df[col].fillna("unknown").astype(str))
        label_encoders[col] = le
        print(f"  编码列 {col}: {len(le.classes_)} 类")

    # --- 特征列构建 ---
    derived_cols = ["byte_ratio", "load_ratio", "pkt_ratio",
                    "dur_rate", "ttl_diff", "avg_pkt_size"]
    # 实际存在的数值列
    available_numeric = [c for c in NUMERIC_COLS if c in train_df.columns]
    feature_cols = available_numeric + derived_cols + CAT_COLS
    # 去重保持顺序
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]
    print(f"  特征维度: {len(feature_cols)}")

    # --- 缺失值填充 ---
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
            if len(test_df) > 0 and col in test_df.columns:
                test_df[col] = test_df[col].fillna(0)

    # --- Z-Score 标准化 ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values).astype(np.float32)
    if len(test_df) > 0:
        X_test = scaler.transform(test_df[feature_cols].values).astype(np.float32)
    else:
        X_test = np.array([])

    # --- 标签构建 ---
    # 10 类标签
    cat_to_id = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    if "attack_cat" in train_df.columns:
        y_train_10class = (
            train_df["attack_cat"]
            .fillna("Normal")
            .astype(str)
            .map(cat_to_id)
            .fillna(0)
            .astype(np.int64)
            .values
        )
        if len(test_df) > 0:
            y_test_10class = (
                test_df["attack_cat"]
                .fillna("Normal")
                .astype(str)
                .map(cat_to_id)
                .fillna(0)
                .astype(np.int64)
                .values
            )
        else:
            y_test_10class = np.array([])
    else:
        y_train_10class = np.zeros(len(train_df), dtype=np.int64)
        y_test_10class = np.zeros(len(test_df), dtype=np.int64) if len(test_df) > 0 else np.array([])

    # 二分类标签 (Normal=0, Attack=1)
    y_train_binary = (y_train_10class != 0).astype(np.int64)
    y_test_binary = (y_test_10class != 0).astype(np.int64) if len(y_test_10class) > 0 else np.array([])

    label_encoders["attack_cat"] = CATEGORY_ORDER

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_10class": y_train_10class,
        "y_test_10class": y_test_10class,
        "y_train_binary": y_train_binary,
        "y_test_binary": y_test_binary,
        "n_features": len(feature_cols),
        "label_encoders": label_encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 数据预处理 (v2)")
    parser.add_argument("--data-dir", default="./data/raw/UNSW-NB15/")
    parser.add_argument("--output-dir", default="./data/processed/")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Edge-IDS 数据预处理 (v2 重构)")
    print("=" * 60)

    # 加载原始数据
    train_df, test_df = load_csv_data(args.data_dir)

    # 编码 + 标准化
    result = encode_and_normalize(train_df, test_df)
    X_train_full = result["X_train"]
    X_test = result["X_test"]
    y_train_10class = result["y_train_10class"]
    y_test_10class = result["y_test_10class"]
    y_train_binary = result["y_train_binary"]
    y_test_binary = result["y_test_binary"]
    n_features = result["n_features"]
    label_encoders = result["label_encoders"]
    scaler = result["scaler"]

    # 类别分布
    print(f"\n  10分类训练集分布:")
    class_counts = np.bincount(y_train_10class, minlength=NUM_CLASSES)
    for i, cat in enumerate(CATEGORY_ORDER):
        if class_counts[i] > 0:
            print(f"    {cat}: {class_counts[i]} ({class_counts[i]/len(y_train_10class)*100:.1f}%)")

    print(f"\n  二分类训练集: Normal={int((y_train_binary==0).sum())}, "
          f"Attack={int((y_train_binary==1).sum())}")

    # 拆分验证集
    (X_train, X_val,
     y_train_10class, y_val_10class,
     y_train_binary, y_val_binary) = train_test_split(
        X_train_full, y_train_10class, y_train_binary,
        test_size=args.val_ratio,
        stratify=y_train_10class,
        random_state=args.seed,
    )

    print(f"\n  拆分后:")
    print(f"    训练集: {X_train.shape[0]:,} 条记录")
    print(f"    验证集: {X_val.shape[0]:,} 条记录")
    if len(X_test) > 0:
        print(f"    测试集: {X_test.shape[0]:,} 条记录")

    # 保存
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "y_train_10class.npy", y_train_10class)
    np.save(output_path / "y_train_binary.npy", y_train_binary)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "y_val_10class.npy", y_val_10class)
    np.save(output_path / "y_val_binary.npy", y_val_binary)

    if len(X_test) > 0:
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_test_10class.npy", y_test_10class)
        np.save(output_path / "y_test_binary.npy", y_test_binary)

    # 保存 Scaler + Encoder
    joblib.dump(scaler, output_path / "scaler.joblib")
    for col, le in label_encoders.items():
        joblib.dump(le, output_path / f"le_{col}.joblib")

    print(f"\n✅ 数据已保存: {output_path}")
    print(f"  特征维度: {n_features}")
    print(f"  类别数: {NUM_CLASSES}")
    print(f"  类别顺序: {CATEGORY_ORDER}")
    print("✅ 预处理完成！运行训练：python train/train.py")


if __name__ == "__main__":
    main()
