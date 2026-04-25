"""
train.py — 主训练脚本

用法：
    python train.py [--data_dir PATH] [--hidden_dim INT] [--activation STR]
                    [--lr FLOAT] [--weight_decay FLOAT] [--batch_size INT]
                    [--epochs INT] [--lr_decay FLOAT] [--decay_step INT]
                    [--model_path STR] [--stats_path STR]
                    [--run_hparam_search]

示例：
    # 使用默认参数训练
    python train.py

    # 自定义参数
    python train.py --hidden_dim 512 --lr 0.01 --epochs 100

    # 先做超参数搜索再训练最优模型
    python train.py --run_hparam_search
"""

import argparse
import numpy as np
import os
import sys

from data_loader import EuroSATDataLoader, get_class_names
from model import MLP
from trainer import Trainer
from evaluator import Evaluator
from visualize import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='EuroSAT MLP 训练脚本')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./EuroSAT_RGB',
                        help='EuroSAT_RGB 数据集目录路径')
    parser.add_argument('--val_ratio',  type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 模型参数
    parser.add_argument('--hidden_dim',   type=int,   default=256,
                        help='隐藏层大小')
    parser.add_argument('--activation',   type=str,   default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='激活函数')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 正则化强度')

    # 训练参数
    parser.add_argument('--lr',         type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--lr_decay',   type=float, default=0.95,
                        help='学习率衰减因子')
    parser.add_argument('--decay_step', type=int,   default=10,
                        help='学习率衰减间隔（epoch）')
    parser.add_argument('--batch_size', type=int,   default=64,
                        help='Mini-batch 大小')
    parser.add_argument('--epochs',     type=int,   default=100,
                        help='训练轮数')

    # 保存路径
    parser.add_argument('--model_path', type=str, default='best_model.npz',
                        help='最优模型权重保存路径')
    parser.add_argument('--stats_path', type=str, default='data_stats.npz',
                        help='归一化统计量保存路径')
    parser.add_argument('--history_path', type=str, default='train_history.npz',
                        help='训练历史保存路径')

    # 超参数搜索
    parser.add_argument('--run_hparam_search', action='store_true',
                        help='是否在训练前进行超参数搜索')
    parser.add_argument('--search_epochs', type=int, default=20,
                        help='超参数搜索时每组的训练轮数')

    # 梯度检验
    parser.add_argument('--grad_check', action='store_true',
                        help='是否进行梯度数值检验（验证反向传播正确性）')

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("  EuroSAT_RGB 手工 MLP 分类器 — 训练脚本")
    print("=" * 60)

    # ── 1. 加载数据 ────────────────────────────────────────────
    print("\n[步骤 1/4] 加载数据集...")
    loader = EuroSATDataLoader(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()
    loader.save_stats(args.stats_path)

    # 保存原始测试集（用于可视化错例）
    np.savez('test_data.npz',
             X_test=X_test, y_test=y_test)

    # ── 2. 梯度检验（可选） ────────────────────────────────────
    if args.grad_check:
        print("\n[可选] 运行梯度数值检验...")
        from model import numerical_gradient_check
        test_model = MLP(input_dim=X_train.shape[1],
                         hidden_dim=args.hidden_dim,
                         output_dim=10,
                         activation=args.activation,
                         weight_decay=args.weight_decay)
        # 取少量样本做检验
        X_small = X_train[:5]
        y_small = y_train[:5]
        max_err = numerical_gradient_check(test_model, X_small, y_small)
        status = "✓ PASS" if max_err < 1e-5 else f"✗ FAIL (err={max_err:.2e})"
        print(f"  梯度检验: {status} (最大相对误差={max_err:.2e})")
        if max_err >= 1e-5:
            print("  警告：梯度检验未通过，反向传播可能有误！")

    # ── 3. 超参数搜索（可选） ──────────────────────────────────
    best_hidden_dim  = args.hidden_dim
    best_lr          = args.lr
    best_weight_decay = args.weight_decay
    best_activation  = args.activation

    if args.run_hparam_search:
        print("\n[步骤 2/4] 超参数搜索（网格搜索）...")
        from hparam_search import grid_search

        search_space = {
            'lr':           [0.05, 0.01, 0.001],
            'hidden_dim':   [128, 256, 512],
            'weight_decay': [0.0, 1e-4, 1e-3],
            'activation':   ['relu', 'tanh'],
        }

        results, best_params = grid_search(
            X_train, y_train, X_val, y_val,
            search_space=search_space,
            epochs=args.search_epochs,
            batch_size=args.batch_size,
            save_path='hparam_results.json',
            verbose=True,
        )

        best_hidden_dim   = best_params['hidden_dim']
        best_lr           = best_params['lr']
        best_weight_decay = best_params['weight_decay']
        best_activation   = best_params['activation']

        print(f"\n最优超参数: {best_params}")
    else:
        print(f"\n[步骤 2/4] 跳过超参数搜索（使用指定参数）")
        print(f"  hidden_dim={best_hidden_dim}, activation={best_activation}, "
              f"lr={best_lr}, weight_decay={best_weight_decay}")

    # ── 4. 使用最优超参数训练完整模型 ─────────────────────────
    print(f"\n[步骤 3/4] 开始训练（{args.epochs} epochs）...")

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=best_hidden_dim,
        output_dim=10,
        activation=best_activation,
        weight_decay=best_weight_decay,
    )
    print(f"模型: {model}")

    trainer = Trainer(
        model=model,
        lr=best_lr,
        lr_decay=args.lr_decay,
        decay_step=args.decay_step,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_path=args.model_path,
        verbose=True,
    )

    history = trainer.train(X_train, y_train, X_val, y_val)

    # 保存训练历史
    np.savez(args.history_path,
             train_loss=history['train_loss'],
             val_loss=history['val_loss'],
             val_acc=history['val_acc'],
             lr_history=history['lr_history'])
    print(f"训练历史已保存至 {args.history_path}")

    # ── 5. 在测试集上快速评估 ─────────────────────────────────
    print(f"\n[步骤 4/4] 加载最优模型并在测试集上评估...")
    model.load_weights(args.model_path)
    evaluator = Evaluator(class_names=get_class_names())
    test_acc, conf_mat = evaluator.evaluate(model, X_test, y_test)
    evaluator.print_report(test_acc, conf_mat)

    # ── 6. 绘制训练曲线 ───────────────────────────────────────
    print("正在保存训练曲线图...")
    plot_training_curves(history, save_path='training_curves.png')
    print("训练曲线已保存至 training_curves.png")

    print("\n训练完成！")
    print(f"  最优模型权重: {args.model_path}")
    print(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")


if __name__ == '__main__':
    main()
