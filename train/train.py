"""
DS-TCN-IDS 两级训练脚本 (v2 重构)

架构:
  Model_A: 二分类 (Normal vs Attack) — 轻量守门员
  Model_B: 十分类 — 攻击子类识别

改进:
  - SMOTE 在原始特征空间过采样 (window 之前)
  - window=10 (攻击信号不被稀释)
  - CrossEntropyLoss + LabelSmoothing (替代 FocalLoss)
  - CosineAnnealingLR

用法:
  python train/train.py                          # 默认 small 模型
  python train/train.py --model-size medium      # GPU高性能
  python train/train.py --skip-preprocess        # 跳过预处理
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ds_tcn_ids import DSTCNIDS, create_model
from train.config import Config, get_tiny_config, get_small_config, get_medium_config


# ============================================================
# 配置常量
# ============================================================

# 10 类标签顺序
CATEGORY_ORDER = [
    "Normal", "Fuzzers", "Analysis", "Backdoor",
    "DoS", "Exploits", "Generic", "Reconnaissance",
    "Shellcode", "Worms",
]
NUM_CLASSES = len(CATEGORY_ORDER)

# 训练超参数 (对齐 edge_ids_v2)
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-5
EARLY_STOP_PATIENCE = 8
LABEL_SMOOTHING = 0.1


# ============================================================
# 数据工具
# ============================================================

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 10):
    """滑动窗口创建时序序列 (窗口标签取尾部)"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, num_classes: int) -> tuple:
    """
    SMOTE 过采样稀有类 (在原始特征空间，window 之前)

    策略:
      - Worms: k=3, → 2000 (极端稀有)
      - Backdoor: → 5000
      - Shellcode: → 5000
      - Analysis: 不做 (模仿 Normal 流量，合成会混淆边界)
    """
    print(f"\n  SMOTE 前各类别样本数:")
    class_counts = np.bincount(y_train, minlength=num_classes)
    for i, cat in enumerate(CATEGORY_ORDER):
        print(f"    {cat}: {class_counts[i]}")

    # --- Worms (ID=9): k=3, 目标 2000 ---
    if class_counts[9] > 1:
        try:
            k = min(3, class_counts[9] - 1)
            smote = SMOTE(sampling_strategy={9: 2000}, k_neighbors=k, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  ✅ SMOTE Worms→2000 (k={k}): {X_train.shape[0]} 样本")
        except Exception as e:
            print(f"  ⚠️ SMOTE Worms 失败: {e}")

    # --- Backdoor (ID=3): → 5000 ---
    if class_counts[3] > 0:
        try:
            smote = SMOTE(sampling_strategy={3: 5000}, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  ✅ SMOTE Backdoor→5000: {X_train.shape[0]} 样本")
        except Exception as e:
            print(f"  ⚠️ SMOTE Backdoor 失败: {e}")

    # --- Shellcode (ID=8): → 5000 ---
    if class_counts[8] > 0:
        try:
            smote = SMOTE(sampling_strategy={8: 5000}, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  ✅ SMOTE Shellcode→5000: {X_train.shape[0]} 样本")
        except Exception as e:
            print(f"  ⚠️ SMOTE Shellcode 失败: {e}")

    # --- Analysis (ID=2): 不做 ---
    print(f"  ℹ️  Analysis(ID=2) 不做 SMOTE（模仿正常流量，避免混淆边界）")

    print(f"\n  SMOTE 后各类别样本数:")
    final_counts = np.bincount(y_train, minlength=num_classes)
    for i, cat in enumerate(CATEGORY_ORDER):
        if final_counts[i] > 0:
            print(f"    {cat}: {final_counts[i]}")

    return X_train, y_train


# ============================================================
# 训练引擎
# ============================================================

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    model_name: str,
    save_path: str,
    epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    use_amp: bool = True,
):
    """统一训练循环：CosineAnnealing + EarlyStopping + 混合精度 + tqdm进度条"""
    best_f1 = 0.0
    best_acc = 0.0
    patience_counter = 0
    all_best_preds, all_best_labels = [], []
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    history = {"train": [], "val": []}

    params, size_mb = model.get_model_size()
    print(f"\n{'='*70}")
    print(f"  {model_name} 训练启动")
    print(f"{'='*70}")
    print(f"  参数: {params:,} | 体积: {size_mb:.2f}MB | 设备: {device}")
    print(f"  Epochs: {epochs} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"{'='*70}\n")

    # 总进度条
    epoch_pbar = tqdm(range(epochs), desc=f"🚀 {model_name}", unit="epoch",
                      ncols=120, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                      "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for epoch in epoch_pbar:
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        scheduler.step()  # CosineAnnealing 逐 epoch 更新

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        epoch_time = time.time() - start_time

        # 指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        lr = scheduler.get_last_lr()[0]

        # 最佳模型
        improved = val_f1 > best_f1
        if improved:
            best_f1, best_acc = val_f1, val_acc
            patience_counter = 0
            all_best_preds, all_best_labels = val_preds, val_labels
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "best_acc": best_acc,
            }, save_path)
        else:
            patience_counter += 1

        # 更新进度条
        epoch_pbar.set_postfix({
            "LR": f"{lr:.2e}",
            "T_Loss": f"{train_loss/len(train_loader):.3f}",
            "V_Loss": f"{val_loss/len(val_loader):.3f}",
            "T_F1": f"{train_f1:.3f}",
            "V_F1": f"{val_f1:.3f}",
            "Best": f"{best_f1:.4f}{' *' if improved else ''}",
        })

        # 每5个epoch或最佳时详细打印
        if (epoch + 1) % 5 == 0 or improved:
            tqdm.write(
                f"  [{model_name}] Epoch {epoch+1:2d}/{epochs} | Time: {epoch_time:.1f}s | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | LR: {lr:.2e}"
                f"{' ✅' if improved else ''}"
            )

        # 记录历史
        history["train"].append({
            "acc": train_acc, "f1": train_f1,
            "loss": train_loss / len(train_loader),
        })
        history["val"].append({
            "acc": val_acc, "f1": val_f1,
            "loss": val_loss / len(val_loader),
        })

        # 早停
        if patience_counter >= patience:
            tqdm.write(f"  ⏹ 早停触发 (patience={patience})")
            break

    epoch_pbar.close()

    # 最终评估
    print(f"\n[{model_name}] 最佳模型评估:")
    cm = confusion_matrix(all_best_labels, all_best_preds)
    print(f"  混淆矩阵:\n{cm}")
    print(f"  最佳准确率: {best_acc:.4f}, 宏平均 F1: {best_f1:.4f}")

    return best_acc, best_f1, history


# ============================================================
# 训练入口
# ============================================================

def train_binary(
    X_train: np.ndarray, y_train_binary: np.ndarray,
    X_val: np.ndarray, y_val_binary: np.ndarray,
    input_dim: int, device: torch.device,
    model_size: str, save_dir: str,
) -> tuple[float, float]:
    """Model_A: 二分类 Normal vs Attack"""

    # 创建序列 (window=10)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train_binary, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val_binary, SEQUENCE_LENGTH)
    print(f"  训练序列: {X_train_seq.shape}, 验证序列: {X_val_seq.shape}")

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train_seq)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_seq), torch.LongTensor(y_val_seq)),
        batch_size=BATCH_SIZE,
    )

    # 模型 (升级容量: DS-TCN 参数少，需更多通道匹配标准卷积的容量)
    model = DSTCNIDS(
        input_dim=input_dim,
        num_classes=2,
        tcn_channels=[128, 256, 256],
        dilations=[1, 2, 4],
        kernel_size=5,
        dropout=0.3,
        use_se_threshold=True,
        use_spatial_branch=True,
        use_gap=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_path = os.path.join(save_dir, "model_binary.pt")
    return run_training(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, "Model_A (二分类)", save_path,
    )


def train_10class(
    X_train: np.ndarray, y_train_10class: np.ndarray,
    X_val: np.ndarray, y_val_10class: np.ndarray,
    input_dim: int, device: torch.device,
    model_size: str, save_dir: str,
) -> tuple[float, float]:
    """Model_B: 十分类攻击子类识别 (含 SMOTE + LabelSmoothing)"""

    # --- SMOTE 在原始特征空间过采样 (window 之前!) ---
    X_train_smote, y_train_smote = apply_smote(X_train, y_train_10class, NUM_CLASSES)

    # --- 创建序列 (window=10) ---
    X_train_seq, y_train_seq = create_sequences(X_train_smote, y_train_smote, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val_10class, SEQUENCE_LENGTH)
    print(f"  训练序列 (SMOTE后): {X_train_seq.shape}, 验证序列: {X_val_seq.shape}")

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train_seq)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_seq), torch.LongTensor(y_val_seq)),
        batch_size=BATCH_SIZE,
    )

    # 模型 (根据 model_size 选择)
    if model_size == "tiny":
        channels = [64, 128, 256]
        dilations = [1, 2, 4]
    elif model_size == "medium":
        channels = [128, 256, 512]
        dilations = [1, 2, 4, 8]
    else:  # small (默认)
        channels = [64, 128, 256, 256]
        dilations = [1, 2, 4, 8]

    model = DSTCNIDS(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        tcn_channels=channels,
        dilations=dilations,
        kernel_size=5,
        dropout=0.3,
        use_se_threshold=True,
        use_spatial_branch=True,
        use_gap=True,
    ).to(device)

    # Label Smoothing CE (替代 FocalLoss — SMOTE 已处理不平衡)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_path = os.path.join(save_dir, "model_10class.pt")
    return run_training(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, "Model_B (10分类)", save_path,
    )


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DS-TCN-IDS 两级训练 (v2)")
    parser.add_argument("--model-size", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--data-path", type=str, default="./data/processed/")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理")
    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n{'='*70}")
    print(f"  DS-TCN-IDS 两级训练 (v2 重构)")
    print(f"{'='*70}")
    print(f"  设备: {gpu_name}")
    print(f"  窗口: {SEQUENCE_LENGTH} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"{'='*70}")

    # --- 预处理 (如果跳过则直接用已处理的) ---
    data_path = Path(args.data_path)

    if not args.skip_preprocess or not (data_path / "X_train.npy").exists():
        print("\n📂 运行预处理...")
        from data.preprocess import main as preprocess_main
        # 模拟命令行参数
        sys.argv = ["preprocess.py", "--output-dir", args.data_path]
        preprocess_main()

    # --- 加载预处理后的数据 ---
    print(f"\n📂 加载数据: {data_path}")
    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    y_train_10class = np.load(data_path / "y_train_10class.npy")
    y_val_10class = np.load(data_path / "y_val_10class.npy")
    y_train_binary = np.load(data_path / "y_train_binary.npy")
    y_val_binary = np.load(data_path / "y_val_binary.npy")

    input_dim = X_train.shape[1]
    print(f"  训练集: {X_train.shape[0]:,} 条记录, {input_dim} 维")
    print(f"  验证集: {X_val.shape[0]:,} 条记录")

    # --- Step 1: 二分类 ---
    print(f"\n{'='*70}")
    print(f"  Step 1/2: Model_A — 二分类 (Normal vs Attack)")
    print(f"{'='*70}")
    acc_a, f1_a, hist_a = train_binary(
        X_train, y_train_binary, X_val, y_val_binary,
        input_dim, device, args.model_size, args.save_dir,
    )

    # --- Step 2: 十分类 ---
    print(f"\n{'='*70}")
    print(f"  Step 2/2: Model_B — 十分类攻击子类识别")
    print(f"{'='*70}")
    acc_b, f1_b, hist_b = train_10class(
        X_train, y_train_10class, X_val, y_val_10class,
        input_dim, device, args.model_size, args.save_dir,
    )

    # --- 最终测试评估 ---
    print(f"\n{'='*70}")
    print(f"  📊 最终测试评估")
    print(f"{'='*70}")

    # 加载最佳 Model_B 并在测试集上评估
    test_data_path = data_path / "X_test.npy"
    if test_data_path.exists():
        X_test = np.load(test_data_path)
        y_test_10class = np.load(data_path / "y_test_10class.npy")
        y_test_binary = np.load(data_path / "y_test_binary.npy")

        # 验证数据来源 (测试集应 ~82K 条记录，非 175K)
        n_test_records = X_test.shape[0]
        print(f"  测试记录: {n_test_records:,} 条")
        if n_test_records > 100000:
            print(f"  ⚠️ 警告: 测试集过大 ({n_test_records:,})，可能是训练数据!")
            print(f"  请重新运行: python data/preprocess.py")

        # 创建测试序列
        X_test_seq, y_test_seq = create_sequences(X_test, y_test_10class, SEQUENCE_LENGTH)
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test_seq)),
            batch_size=BATCH_SIZE,
        )
        print(f"  测试序列: {X_test_seq.shape}")

        # 加载最佳 Model_B
        channels = [64, 128, 256, 256] if args.model_size == "small" else (
            [32, 64, 128] if args.model_size == "tiny" else [128, 256, 512]
        )
        dilations = [1, 2, 4, 8] if args.model_size != "tiny" else [1, 2, 4]
        if args.model_size == "tiny":
            channels = [32, 64, 128]
            dilations = [1, 2, 4]
        elif args.model_size == "medium":
            channels = [128, 256, 512, 512]
            dilations = [1, 2, 4, 8]
        else:
            channels = [64, 128, 256, 256]
            dilations = [1, 2, 4, 8]

        model_b = DSTCNIDS(
            input_dim=input_dim,
            num_classes=NUM_CLASSES,
            tcn_channels=channels,
            dilations=dilations,
            kernel_size=5,
            dropout=0.3,
            use_se_threshold=True,
            use_spatial_branch=True,
            use_gap=True,
        ).to(device)

        checkpoint = torch.load(
            os.path.join(args.save_dir, "model_10class.pt"),
            map_location=device,
            weights_only=False,
        )
        model_b.load_state_dict(checkpoint["model_state_dict"])
        model_b.eval()

        # 评估
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model_b(batch_x.to(device))
                test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                test_labels.extend(batch_y.numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        params, size_mb = model_b.get_model_size()

        print(f"  Model_A 二分类 F1: {f1_a:.4f}")
        print(f"  Model_B 十分类 F1: {test_f1:.4f}")
        print(f"  准确率:      {test_acc:.4f}")
        print(f"  宏平均 F1:   {test_f1:.4f}")
        print(f"  模型参数:    {params:,}")
        print(f"  模型体积:    {size_mb:.2f} MB (FP32)")

        print(f"\n  逐类分类报告:")
        print(classification_report(test_labels, test_preds,
              target_names=CATEGORY_ORDER, zero_division=0, digits=4))

    # 汇总
    print(f"\n{'='*70}")
    print(f"  训练完成汇总")
    print(f"{'='*70}")
    print(f"  Model_A 二分类 → F1: {f1_a:.4f}")
    print(f"  Model_B 十分类 → F1: {f1_b:.4f} (验证集)")
    if test_data_path.exists():
        print(f"  测试集十分类 → F1: {test_f1:.4f}")
    print(f"  模型已保存: {args.save_dir}/")


if __name__ == "__main__":
    main()
