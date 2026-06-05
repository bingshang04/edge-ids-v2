"""
DS-TCN-IDS 训练脚本 (GPU + tqdm 进度条)

用法:
  python train/train.py                          # 默认 small 模型
  python train/train.py --model-size tiny         # tiny 模型 (树莓派部署)
  python train/train.py --model-size medium       # medium 模型 (GPU高性能)
  python train/train.py --ablation               # 消融实验
  python train/train.py --resume checkpoints/best_model.pt  # 断点续训

环境: conda activate edge (PyTorch 2.11 + CUDA 12.8)

参考论文整合:
  - 论文1 (Nazre): TCN 基准
  - 论文2 (赵建): Focal Loss + SE 软阈值 + 并联时空融合
  - 论文5 (顾兆军): 深度可分离卷积 + GAP
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm  # 进度条

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ds_tcn_ids import DSTCNIDS, create_model
from models.focal_loss import FocalLoss
from train.config import Config, get_tiny_config, get_small_config, get_medium_config


# ============================================================
# 训练器 (tqdm 进度条版)
# ============================================================


class Trainer:
    """DS-TCN-IDS GPU 训练器"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        save_dir: str = "./checkpoints",
        use_amp: bool = True,  # 混合精度训练
        train_labels: np.ndarray | None = None,  # 用于预计算Focal Loss alpha
    ):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp

        # 设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
            self.use_amp = False

        self.model = self.model.to(self.device)

        # 损失函数 — 预计算 alpha (核心修复!)
        if config.train.use_focal_loss:
            if train_labels is not None:
                # 从全量训练标签预计算类别权重 (论文2公式4)
                alpha = FocalLoss.compute_class_weights(
                    torch.LongTensor(train_labels),
                    num_classes=model.num_classes,
                )
                print(f"  预计算 alpha: {alpha.tolist()}")
            else:
                alpha = None
            self.criterion = FocalLoss(
                alpha=alpha,
                gamma=config.train.focal_gamma,
                reduction="mean",
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 优化器 (增大 weight_decay 防过拟合)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        # 学习率调度 — 在 fit() 中根据数据量动态创建 OneCycleLR
        self.scheduler = None

        # 混合精度
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # 训练状态
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {"train": [], "val": []}

    def train_epoch(self, train_loader: DataLoader, epoch_pbar: tqdm) -> dict:
        """训练一个 epoch (带进度条)"""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        batch_pbar = tqdm(
            train_loader,
            desc=f"  🔄 训练",
            leave=False,
            unit="batch",
            ncols=100,
        )

        for data, targets in batch_pbar:
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # 更高效的清零

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # OneCycleLR 逐batch更新
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # 更新进度条
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "loss": total_loss / len(train_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }
        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> tuple[dict, np.ndarray, np.ndarray]:
        """验证/测试评估"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        val_pbar = tqdm(val_loader, desc="  📊 验证", leave=False, unit="batch", ncols=100)

        for data, targets in val_pbar:
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }
        return metrics, all_preds, all_labels

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """完整训练循环 (带总进度条)"""
        # 打印训练配置
        params, size_mb = self.model.get_model_size()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        print(f"\n{'='*70}")
        print(f"  DS-TCN-IDS 训练启动")
        print(f"{'='*70}")
        print(f"  设备:      {gpu_name}")
        print(f"  混合精度:  {'✅' if self.use_amp else '❌'}")
        print(f"  模型参数:  {params:,}")
        print(f"  模型体积:  {size_mb:.2f} MB (FP32)")
        print(f"  学习率:    {self.config.train.learning_rate}")
        print(f"  Batch:     {self.config.train.batch_size}")
        print(f"  Focal Loss: γ={self.config.train.focal_gamma}")
        print(f"  SE 软阈值: {'✅' if self.model.use_se_threshold else '❌'}")
        print(f"  空间分支:  {'✅' if self.model.use_spatial_branch else '❌'}")
        print(f"  GAP 模式:  {'✅' if self.model.use_gap else '❌'}")
        print(f"{'='*70}\n")

        # OneCycleLR: 先升后降，帮助跳出"全预测Normal"的局部最优
        total_steps = len(train_loader) * self.config.train.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.train.learning_rate * 3,  # 峰值 lr = 0.003
            total_steps=total_steps,
            pct_start=0.3,  # 前30%步数升温
            anneal_strategy="cos",
            final_div_factor=1e4,  # 最终 lr = 0.003 / 10000 = 3e-7
        )

        # 总进度条
        epoch_pbar = tqdm(
            range(self.config.train.epochs),
            desc="🚀 总进度",
            unit="epoch",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for epoch in epoch_pbar:
            self.current_epoch = epoch
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch_pbar)

            # 验证
            val_metrics, _, _ = self.evaluate(val_loader)

            # OneCycleLR 已在 train_epoch 中逐batch更新，此处无需额外操作

            epoch_time = time.time() - start_time

            # 记录
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            # 早停检查
            improved = False
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch
                self.patience_counter = 0
                improved = True
                self.save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1

            # 更新进度条描述
            lr = self.optimizer.param_groups[0]["lr"]
            epoch_pbar.set_postfix({
                "LR": f"{lr:.2e}",
                "T_Loss": f"{train_metrics['loss']:.3f}",
                "V_Loss": f"{val_metrics['loss']:.3f}",
                "T_F1": f"{train_metrics['f1']:.3f}",
                "V_F1": f"{val_metrics['f1']:.3f}",
                "Best": f"{self.best_val_f1:.4f}{' *' if improved else ''}",
            })

            # 每10个epoch打印详细指标
            if (epoch + 1) % 10 == 0 or improved:
                tqdm.write(
                    f"  Epoch {epoch+1:3d}/{self.config.train.epochs} | "
                    f"LR: {lr:.6f} | Time: {epoch_time:.1f}s | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f}"
                    f"{' ✅' if improved else ''}"
                )

            # 早停
            if self.patience_counter >= self.config.train.early_stop_patience:
                tqdm.write(f"\n⏹ 早停触发 (patience={self.config.train.early_stop_patience})")
                break

        epoch_pbar.close()
        return self.history

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = self.save_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_f1": self.best_val_f1,
                "config": self.config,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        path = Path(filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        self.current_epoch = checkpoint.get("epoch", 0)
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        print(f"📂 加载检查点: {path} (Epoch {self.current_epoch+1}, F1={self.best_val_f1:.4f})")


# ============================================================
# 消融实验
# ============================================================


def run_ablation_experiment(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    base_config: Config,
):
    """
    消融实验: 逐步关闭各模块，量化贡献

    输出对比表格，直接可用于论文
    """
    print(f"\n{'='*70}")
    print("  消融实验 — 量化各模块贡献")
    print(f"{'='*70}\n")

    train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels))
    val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.LongTensor(val_labels))
    train_loader = DataLoader(train_dataset, batch_size=base_config.train.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=base_config.train.batch_size)

    experiments = [
        # (名称, SE, 空间分支, FocalLoss, GAP)
        ("🔥 完整 DS-TCN-IDS",     True,  True,  True,  True),
        ("   - SE 自适应软阈值",     False, True,  True,  True),
        ("   - 并联空间分支",       True,  False, True,  True),
        ("   - Focal Loss (→CE)",   True,  True,  False, True),
        ("   - GAP (→全连接)",      True,  True,  True,  False),
        ("📏 纯标准 TCN (基线)",    False, False, False, False),
    ]

    results = []
    for name, se, spatial, focal, gap in experiments:
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        config = base_config
        config.model.use_se_threshold = se
        config.model.use_spatial_branch = spatial
        config.train.use_focal_loss = focal
        config.model.use_gap = gap

        model = DSTCNIDS(
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes,
            tcn_channels=config.model.tcn_channels,
            dilations=config.model.dilations,
            use_se_threshold=se,
            use_spatial_branch=spatial,
            use_gap=gap,
        )

        trainer = Trainer(model, config, save_dir=f"./checkpoints/ablation_{name.strip()[:20]}", train_labels=train_labels)
        history = trainer.fit(train_loader, val_loader)

        best_idx = np.argmax([h["f1"] for h in history["val"]])
        best = history["val"][best_idx]
        params, size_mb = model.get_model_size()

        results.append({
            "name": name.strip(),
            "params": params,
            "size_mb": round(size_mb, 3),
            "epoch": best_idx + 1,
            "acc": best["accuracy"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
        })

    # 打印结果表
    print(f"\n{'='*90}")
    print("  消融实验结果汇总")
    print(f"{'='*90}")
    print(f"{'实验组':<30} {'准确率':>7} {'F1':>7} {'精确率':>7} {'召回率':>7} {'参数量':>9} {'体积':>7}")
    print(f"{'─'*90}")

    baseline = results[-1]  # 纯标准TCN
    for r in results:
        acc_delta = r["acc"] - baseline["acc"]
        f1_delta = r["f1"] - baseline["f1"]
        delta_str = f"(+{acc_delta:+.2%})" if r != results[-1] else "(基线)"
        print(
            f"{r['name']:<30} {r['acc']:>7.4f} {r['f1']:>7.4f} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
            f"{r['params']:>8,} {r['size_mb']:>5.1f}MB {delta_str}"
        )

    # 保存为 JSON
    with open("./checkpoints/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 结果已保存: ./checkpoints/ablation_results.json")

    return results


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="DS-TCN-IDS 训练 🚀",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train/train.py                              # 默认 small 模型
  python train/train.py --model-size tiny             # 极致轻量 (树莓派5)
  python train/train.py --model-size medium --gpu 0   # 高性能GPU
  python train/train.py --ablation                    # 消融实验
  python train/train.py --resume checkpoints/best_model.pt  # 断点续训
        """,
    )
    # 模型
    parser.add_argument("--model-size", default="small", choices=["tiny", "small", "medium"])
    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    # 数据
    parser.add_argument("--data-path", type=str, default="./data/processed/")
    parser.add_argument("--no-focal-loss", action="store_true")
    # 实验
    parser.add_argument("--ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--resume", type=str, default=None, help="断点续训路径")
    # 系统
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 配置
    config_map = {"tiny": get_tiny_config, "small": get_small_config, "medium": get_medium_config}
    config = config_map[args.model_size]()
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.use_focal_loss = not args.no_focal_loss

    # 加载数据
    data_path = Path(args.data_path)
    print(f"\n📂 加载数据: {data_path}")

    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    y_val = np.load(data_path / "y_val.npy")
    X_test = np.load(data_path / "X_test.npy") if (data_path / "X_test.npy").exists() else None
    y_test = np.load(data_path / "y_test.npy") if (data_path / "y_test.npy").exists() else None

    # 更新配置以匹配数据
    config.model.input_dim = X_train.shape[2]
    num_classes = len(np.unique(np.concatenate([y_train, y_val])))
    config.model.num_classes = num_classes

    print(f"  训练集: {X_train.shape}, 类别: {np.bincount(y_train)}")
    print(f"  验证集: {X_val.shape}, 类别: {np.bincount(y_val)}")
    if X_test is not None:
        print(f"  测试集: {X_test.shape}, 类别: {np.bincount(y_test)}")
    print(f"  特征维度: {config.model.input_dim}, 类别数: {num_classes}")

    # 创建模型
    model = DSTCNIDS(
        input_dim=config.model.input_dim,
        num_classes=num_classes,
        tcn_channels=config.model.tcn_channels,
        dilations=config.model.dilations,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        use_se_threshold=config.model.use_se_threshold,
        use_spatial_branch=config.model.use_spatial_branch,
        use_gap=config.model.use_gap,
    )

    # 数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    # 稀有类数据增强: 对样本数 < 300 的类别添加高斯噪声扩充样本
    # 解决 Worms(17个)/Shellcode(184个) 等极端稀有类学习不足问题
    minority_threshold = 300
    class_counts = np.bincount(y_train)
    aug_X_list, aug_y_list = [], []
    rng = np.random.RandomState(args.seed)

    for cls_idx in range(num_classes):
        n_cls = class_counts[cls_idx]
        if 0 < n_cls < minority_threshold:
            mask = y_train == cls_idx
            cls_data = X_train[mask]  # (n_cls, seq_len, features)
            # 每个样本生成 8 个噪声变体 (σ=0.02, 约2%的标准化特征扰动)
            aug_factor = 8
            noise_std = 0.02
            noise = rng.randn(n_cls * aug_factor, *cls_data.shape[1:]).astype(np.float32) * noise_std
            aug_data = np.tile(cls_data, (aug_factor, 1, 1)) + noise
            aug_X_list.append(aug_data)
            aug_y_list.append(np.full(n_cls * aug_factor, cls_idx, dtype=np.int64))
            print(f"  🔄 增强类别 {cls_idx}: {n_cls} → {n_cls + n_cls * aug_factor} 样本 (+{aug_factor}×噪声扩充)")

    if aug_X_list:
        X_train = np.concatenate([X_train] + aug_X_list)
        y_train = np.concatenate([y_train] + aug_y_list)
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        print(f"  增强后训练集: {X_train.shape}, 类别: {np.bincount(y_train)}")
        # 重建 dataset
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))

    # 标准 shuffle 加载 (不用 WeightedRandomSampler, 避免重复采样过拟合)
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,  # 保留最后不完整batch（含稀有类）
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )

    # 消融实验 or 标准训练
    if args.ablation:
        run_ablation_experiment(X_train, y_train, X_val, y_val, config)
    else:
        trainer = Trainer(
            model, config,
            save_dir=args.save_dir,
            use_amp=not args.no_amp,
            train_labels=y_train,  # 预计算 Focal Loss alpha
        )

        # 断点续训
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # 训练
        history = trainer.fit(train_loader, val_loader)

        # 最终测试评估
        test_loader = val_loader if X_test is None else DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=config.train.batch_size, num_workers=args.num_workers,
        )
        test_metrics, y_pred, y_true = trainer.evaluate(test_loader)

        print(f"\n{'='*70}")
        print(f"  📊 最终测试评估")
        print(f"{'='*70}")
        print(f"  最佳Epoch:   {trainer.best_epoch + 1}")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Precision:   {test_metrics['precision']:.4f} (macro)")
        print(f"  Recall:      {test_metrics['recall']:.4f} (macro)")
        print(f"  F1-Score:    {test_metrics['f1']:.4f} (macro)")
        print(f"  模型参数:    {model.get_model_size()[0]:,}")
        print(f"  模型体积:    {model.get_model_size()[1]:.2f} MB (FP32)")

        # 每类详细报告
        print(f"\n  逐类分类报告:")
        print(classification_report(y_true, y_pred, zero_division=0, digits=4))

        # 保存最终结果
        results = {
            "model_size": args.model_size,
            "best_epoch": trainer.best_epoch + 1,
            "train_best": trainer.history["train"][trainer.best_epoch],
            "val_best": trainer.history["val"][trainer.best_epoch],
            "test": {k: float(v) for k, v in test_metrics.items()},
            "num_params": model.get_model_size()[0],
            "model_size_mb": model.get_model_size()[1],
        }
        with open(trainer.save_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n📁 结果已保存: {trainer.save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
