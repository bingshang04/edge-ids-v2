"""
Edge-IDS v2.0 统一训练脚本（ECA-TCN + UNSW-NB15 多分类）
训练完成后自动保存模型、Scaler 和 LabelEncoder

多分类: 1 正常 + 9 攻击 = 10 类
攻击类别: Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging

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
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-5

# ECA-TCN 模型参数
NUM_CLASSES = 10        # 多分类: 1 Normal + 9 Attack
NUM_CHANNELS = [128, 256, 256]
KERNEL_SIZE = 5
DROPOUT = 0.3
USE_ECA = True


# ====================== Focal Loss（处理类别不平衡）======================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    对难分类样本和少数类样本加权，解决 UNSW-NB15 的严重类别不平衡问题。
    gamma=2 时，pt=0.9 的易分类样本权重降低 100x。
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 各类别权重 tensor，shape (num_classes,)，或 None
            gamma: 聚焦参数，越大越关注难分类样本
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) — 模型 logits
            targets: (batch,) — 类别标签
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 每个样本的 pt
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ====================== 数据加载与特征工程 ======================
def load_and_preprocess():
    """加载 UNSW-NB15 数据 + 48维特征工程 + 多分类标签编码 + 保存预处理器"""
    logger.info("加载数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    logger.info(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")

    # 基础数值特征（按 CSV 列自然顺序，排除非数值列）
    numeric_cols = [col for col in train_df.columns
                    if col not in ['id', 'proto', 'service', 'state', 'attack_cat', 'label']]

    # 衍生特征
    for df in [train_df, test_df]:
        df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1e-6)
        df['load_ratio'] = df['sload'] / (df['dload'] + 1e-6)
        df['pkt_ratio'] = df['spkts'] / (df['spkts'] + df['dpkts'] + 1e-6)
        df['dur_rate'] = df['dur'] * df['rate']
        df['ttl_diff'] = df['sttl'] - df['dttl']
        df['avg_pkt_size'] = (df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts'] + 1e-6)

    # 类别特征编码
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

    # attack_cat 编码（多分类标签 0-9）
    all_attack_cats = pd.concat([train_df['attack_cat'], test_df['attack_cat']], axis=0).fillna('Normal').astype(str)
    le_attack = LabelEncoder()
    le_attack.fit(all_attack_cats)
    y_train = le_attack.transform(train_df['attack_cat'].fillna('Normal').astype(str))
    y_test = le_attack.transform(test_df['attack_cat'].fillna('Normal').astype(str))
    label_encoders['attack_cat'] = le_attack
    logger.info(f"attack_cat 编码: {len(le_attack.classes_)} 类 → {list(le_attack.classes_)}")
    logger.info(f"各类别样本数:\n{pd.Series(y_train).value_counts().sort_index().to_dict()}")

    # 最终48维特征列表
    feature_cols = numeric_cols + ['byte_ratio', 'load_ratio', 'pkt_ratio',
                                   'dur_rate', 'ttl_diff', 'avg_pkt_size'] + cat_cols
    logger.info(f"特征维度: {len(feature_cols)}")

    # 缺失值处理
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    total_train = len(y_train)
    normal_count = (y_train == 0).sum()
    attack_count = total_train - normal_count
    logger.info(f"正常样本: {normal_count} ({normal_count / total_train:.2%}), "
                f"攻击样本: {attack_count} ({attack_count / total_train:.2%})")

    return X_train, X_test, y_train, y_test, len(feature_cols), scaler, label_encoders


def create_sequences(X, y, seq_length):
    """滑动窗口创建时序序列"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)


# ====================== 训练 ======================
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"设备: {device}")

    # 加载数据
    X_train, X_test, y_train, y_test, input_dim, scaler, label_encoders = load_and_preprocess()

    # 时序序列
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQUENCE_LENGTH)
    logger.info(f"训练序列: {X_train_seq.shape}, 测试序列: {X_test_seq.shape}")

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train_seq)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test_seq)),
        batch_size=BATCH_SIZE
    )

    # ECA-TCN 模型（多分类）
    model = TCN(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        use_eca=USE_ECA,
    ).to(device)
    logger.info(f"ECA-TCN 参数量: {model.num_params / 1e6:.2f}M")
    logger.info(f"模型信息: {model.get_model_info()}")

    # Focal Loss + 类别权重（对少数攻击类额外加权）
    # 类别: 0=Normal,1=Analysis,2=Backdoor,3=DoS,4=Exploits,5=Fuzzers,6=Generic,7=Reconnaissance,8=Shellcode,9=Worms
    # 对 Worms(9), Shellcode(8), Backdoor(2), Analysis(1) 额外加权 3-5x
    class_weights = np.ones(NUM_CLASSES)
    # 根据 UNSW-NB15 的严重不平衡，给稀有攻击类加权重
    rare_attack_classes = {1: 4.0, 2: 4.0, 8: 5.0, 9: 5.0}  # Analysis, Backdoor, Shellcode, Worms
    for cls_idx, weight_mult in rare_attack_classes.items():
        class_weights[cls_idx] *= weight_mult
    # 一般攻击类
    class_weights[3] = 2.0   # DoS
    class_weights[4] = 2.0   # Exploits
    class_weights[5] = 2.5   # Fuzzers
    class_weights[6] = 2.5   # Generic
    class_weights[7] = 1.5   # Reconnaissance
    # Normal 保持 1.0

    alpha = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 训练循环
    best_acc, best_f1 = 0.0, 0.0
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

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x.to(device))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        logger.info(f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Loss: {train_loss / len(train_loader):.4f} | "
                    f"Acc: {acc:.4f} | P_macro: {prec:.4f} | R_macro: {rec:.4f} | F1_macro: {f1:.4f}")

        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_acc, best_f1 = acc, f1
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'tcn_model_3.0.pth'))
            logger.info(f"  → 保存最佳模型 (Acc: {acc:.4f}, F1_macro: {f1:.4f})")

    # 最终评估
    logger.info("\n最终评估:")
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"混淆矩阵:\n{cm}")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    logger.info("各类别准确率:")
    le_attack = label_encoders['attack_cat']
    for i, name in enumerate(le_attack.classes_):
        logger.info(f"  {name}: {per_class_acc[i]:.4f}" if i < len(per_class_acc) else f"  {name}: N/A")
    logger.info(f"总体准确率: {best_acc:.4f}, 宏平均 F1: {best_f1:.4f}")

    # 保存预处理器
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'unsw_scaler3.0.joblib'))
    logger.info(f"Scaler 已保存 → {MODEL_DIR}/unsw_scaler3.0.joblib")
    for col, le in label_encoders.items():
        joblib.dump(le, os.path.join(MODEL_DIR, f'le_{col}.joblib'))
        logger.info(f"LabelEncoder({col}) 已保存")

    logger.info("\n训练完成！")


if __name__ == "__main__":
    train()
