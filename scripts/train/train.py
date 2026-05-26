"""
Edge-IDS v2.0 两步级联训练脚本（ECA-TCN）
Model_A: 二分类 [64,128] Normal vs Attack
Model_B: 九分类 [128,256] 8种攻击子类识别
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging
from imblearn.over_sampling import SMOTE

from src.models.tcn_model import TCN

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ====================== 配置 ======================
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(ROOT_DIR, 'data', 'models')

TRAIN_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
TEST_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')

SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-5
EARLY_STOP_PATIENCE = 8

KERNEL_SIZE = 5
DROPOUT = 0.3
USE_ECA = True

# 9 类（DoS + Exploits 合并）
CATEGORY_ORDER = ['Normal', 'Analysis', 'Backdoor', 'DoS_Exploits', 'Fuzzers',
                  'Generic', 'Reconnaissance', 'Shellcode', 'Worms']


# ====================== 公用数据加载 ======================
def _load_raw_data():
    """加载原始 CSV 并完成特征工程/标准化，返回 X_train/X_test 和原始 DataFrame"""
    logger.info("加载数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    logger.info(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")

    numeric_cols = [col for col in train_df.columns
                    if col not in ['id', 'proto', 'service', 'state', 'attack_cat', 'label']]

    for df in [train_df, test_df]:
        df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1e-6)
        df['load_ratio'] = df['sload'] / (df['dload'] + 1e-6)
        df['pkt_ratio'] = df['spkts'] / (df['spkts'] + df['dpkts'] + 1e-6)
        df['dur_rate'] = df['dur'] * df['rate']
        df['ttl_diff'] = df['sttl'] - df['dttl']
        df['avg_pkt_size'] = (df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts'] + 1e-6)

    cat_cols = ['proto', 'service', 'state']
    label_encoders = {}
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0).fillna('unknown').astype(str)
        le = LabelEncoder()
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].fillna('unknown').astype(str))
        test_df[col] = le.transform(test_df[col].fillna('unknown').astype(str))
        label_encoders[col] = le
        logger.info(f"编码列 {col}: {len(le.classes_)} 类")

    feature_cols = numeric_cols + ['byte_ratio', 'load_ratio', 'pkt_ratio',
                                   'dur_rate', 'ttl_diff', 'avg_pkt_size'] + cat_cols
    logger.info(f"特征维度: {len(feature_cols)}")

    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    # 9 类标签映射（DoS + Exploits 合并为 DoS_Exploits）
    cat_to_id = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    cat_to_id['DoS'] = cat_to_id['DoS_Exploits']
    cat_to_id['Exploits'] = cat_to_id['DoS_Exploits']

    y_train_9class = train_df['attack_cat'].fillna('Normal').astype(str).map(cat_to_id).values
    y_test_9class = test_df['attack_cat'].fillna('Normal').astype(str).map(cat_to_id).values

    # 二分类标签（Normal=0, Attack=1）
    y_train_binary = (y_train_9class != 0).astype(int)
    y_test_binary = (y_test_9class != 0).astype(int)

    label_encoders['attack_cat'] = CATEGORY_ORDER
    logger.info(f"attack_cat 编码: {len(CATEGORY_ORDER)} 类 → {CATEGORY_ORDER}")
    logger.info(f"9分类样本分布:\n{pd.Series(y_train_9class).value_counts().sort_index().to_dict()}")
    logger.info(f"二分类样本: Normal={int((y_train_binary==0).sum())}, Attack={int((y_train_binary==1).sum())}")

    return (X_train, X_test, y_train_binary, y_test_binary,
            y_train_9class, y_test_9class, len(feature_cols), scaler, label_encoders)


def _apply_smote_9class(X_train, y_train_9class):
    """对九分类训练集做 SMOTE（Worms k=3→2000, Backdoor→5000, Shellcode→5000, Analysis 不做）"""
    logger.info(f"\nSMOTE 前各类别样本数:\n{pd.Series(y_train_9class).value_counts().sort_index().to_dict()}")

    # Worms (ID=8): k=3, 目标 2000
    try:
        worms_count = int((y_train_9class == 8).sum())
        k = min(3, worms_count - 1) if worms_count > 1 else 1
        smote_worms = SMOTE(sampling_strategy={8: 2000}, k_neighbors=k, random_state=42)
        X_train, y_train_9class = smote_worms.fit_resample(X_train, y_train_9class)
        logger.info(f"SMOTE Worms→2000 (k={k}): {X_train.shape[0]} 样本")
    except Exception as e:
        logger.warning(f"SMOTE Worms 失败: {e}")

    # Backdoor (ID=2): → 5000
    try:
        smote_backdoor = SMOTE(sampling_strategy={2: 5000}, random_state=42)
        X_train, y_train_9class = smote_backdoor.fit_resample(X_train, y_train_9class)
        logger.info(f"SMOTE Backdoor→5000: {X_train.shape[0]} 样本")
    except Exception as e:
        logger.warning(f"SMOTE Backdoor 失败: {e}")

    # Shellcode (ID=7): → 5000
    try:
        smote_shellcode = SMOTE(sampling_strategy={7: 5000}, random_state=42)
        X_train, y_train_9class = smote_shellcode.fit_resample(X_train, y_train_9class)
        logger.info(f"SMOTE Shellcode→5000: {X_train.shape[0]} 样本")
    except Exception as e:
        logger.warning(f"SMOTE Shellcode 失败: {e}")

    # Analysis (ID=1) 不做 SMOTE（模仿正常流量，合成会混淆边界）
    logger.info(f"Analysis(ID=1) 不做 SMOTE（模仿正常流量特征，避免混淆 Normal 边界）")

    logger.info(f"SMOTE 后各类别样本数:\n{pd.Series(y_train_9class).value_counts().sort_index().to_dict()}\n")
    return X_train, y_train_9class


# ====================== 公用训练引擎 ======================
def _run_training_loop(model, train_loader, test_loader, criterion, optimizer,
                       scheduler, device, model_name, model_save_path):
    """统一训练循环：CosineAnnealing + EarlyStopping"""
    best_f1, best_acc = 0.0, 0.0
    patience_counter = 0
    all_preds, all_labels = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        epoch_preds, epoch_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x.to(device))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                epoch_preds.extend(preds)
                epoch_labels.extend(batch_y.numpy())

        acc = accuracy_score(epoch_labels, epoch_preds)
        f1 = f1_score(epoch_labels, epoch_preds, average='macro', zero_division=0)
        lr = scheduler.get_last_lr()[0]

        logger.info(f"[{model_name}] Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Loss: {train_loss / len(train_loader):.4f} | "
                    f"Acc: {acc:.4f} | F1_macro: {f1:.4f} | LR: {lr:.2e}")

        # 最佳模型保存
        if f1 > best_f1:
            best_f1, best_acc = f1, acc
            patience_counter = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"  → 保存最佳模型 → {model_save_path}")
            all_preds, all_labels = epoch_preds, epoch_labels
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"  → EarlyStopping (patience={EARLY_STOP_PATIENCE})")
                break

    # 最终评估
    if not all_preds:
        all_preds, all_labels = epoch_preds, epoch_labels
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\n[{model_name}] 最佳模型评估:")
    logger.info(f"混淆矩阵:\n{cm}")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    logger.info(f"各类别准确率: {dict(enumerate(per_class_acc.round(4)))}")
    logger.info(f"总体准确率: {best_acc:.4f}, 宏平均 F1: {best_f1:.4f}")

    return best_acc, best_f1


# ====================== 训练入口 ======================
def train_binary(X_train, X_test, y_train_binary, y_test_binary, input_dim, device):
    """Model_A: 二分类 Normal vs Attack，轻量 [64,128]"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Model_A: 二分类训练 (Normal vs Attack)")
    logger.info(f"{'='*50}")

    X_train_seq, y_train_seq = create_sequences(X_train, y_train_binary, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test_binary, SEQUENCE_LENGTH)
    logger.info(f"训练序列: {X_train_seq.shape}, 测试序列: {X_test_seq.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train_seq)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test_seq)),
        batch_size=BATCH_SIZE
    )

    model = TCN(input_dim=input_dim, num_classes=2, num_channels=[64, 128],
                kernel_size=KERNEL_SIZE, dropout=DROPOUT, use_eca=USE_ECA).to(device)
    logger.info(f"参数量: {model.num_params / 1e6:.2f}M, channels=[64,128]")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_path = os.path.join(MODEL_DIR, 'tcn_model_binary.pth')
    return _run_training_loop(model, train_loader, test_loader, criterion, optimizer,
                              scheduler, device, 'Model_A', save_path)


def train_9class(X_train, X_test, y_train_9class, y_test_9class, input_dim, device):
    """Model_B: 九分类攻击子类识别，[128,256]，Label Smoothing CE"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Model_B: 九分类训练 ({' / '.join(CATEGORY_ORDER[1:])})")
    logger.info(f"{'='*50}")

    # SMOTE 仅对攻击样本（排除 Normal 由数据工程师控制）
    X_train_smote, y_train_smote = _apply_smote_9class(X_train, y_train_9class)

    X_train_seq, y_train_seq = create_sequences(X_train_smote, y_train_smote, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test_9class, SEQUENCE_LENGTH)
    logger.info(f"训练序列 (SMOTE后): {X_train_seq.shape}, 测试序列: {X_test_seq.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train_seq)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test_seq)),
        batch_size=BATCH_SIZE
    )

    model = TCN(input_dim=input_dim, num_classes=len(CATEGORY_ORDER),
                num_channels=[128, 256], kernel_size=KERNEL_SIZE,
                dropout=DROPOUT, use_eca=USE_ECA).to(device)
    logger.info(f"参数量: {model.num_params / 1e6:.2f}M, channels=[128,256]")

    # Label Smoothing CE（替代 FocalLoss）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_path = os.path.join(MODEL_DIR, 'tcn_model_9class.pth')
    return _run_training_loop(model, train_loader, test_loader, criterion, optimizer,
                              scheduler, device, 'Model_B', save_path)


def create_sequences(X, y, seq_length):
    """滑动窗口创建时序序列"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)


# ====================== 主入口 ======================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"设备: {device} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")

    # 一次性加载数据，拆分标签
    (X_train, X_test, y_train_binary, y_test_binary,
     y_train_9class, y_test_9class, input_dim, scaler, label_encoders) = _load_raw_data()

    # Step 1: 二分类
    train_binary(X_train, X_test, y_train_binary, y_test_binary, input_dim, device)

    # Step 2: 九分类
    train_9class(X_train, X_test, y_train_9class, y_test_9class, input_dim, device)

    # 保存预处理器（两个模型共享同一套）
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'unsw_scaler3.0.joblib'))
    logger.info(f"Scaler 已保存 → {MODEL_DIR}/unsw_scaler3.0.joblib")
    for col, le in label_encoders.items():
        joblib.dump(le, os.path.join(MODEL_DIR, f'le_{col}.joblib'))
        logger.info(f"LabelEncoder({col}) 已保存")

    logger.info("\n全部训练完成！")
