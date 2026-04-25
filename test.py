"""
test.py — 主测试脚本

功能：
1. 加载训练好的最优模型权重
2. 在独立测试集上评估分类准确率
3. 打印混淆矩阵
4. 调用可视化：权重图像 + 错例分析

用法：
    python test.py [--model_path PATH] [--stats_path PATH]
                   [--data_dir PATH] [--hidden_dim INT]
                   [--activation STR] [--skip_visualize]

示例：
    # 使用默认路径
    python test.py

    # 指定模型路径
    python test.py --model_path best_model.npz --hidden_dim 512 --activation relu
"""

import argparse
import numpy as np
import os

from data_loader import EuroSATDataLoader, get_class_names
from model import MLP
from evaluator import Evaluator
from visualize import (
    plot_training_curves,
    visualize_weights,
    visualize_error_samples,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description='EuroSAT MLP 测试脚本')

    parser.add_argument('--model_path', type=str, default='best_model.npz',
                        help='训练好的模型权重文件路径')
    parser.add_argument('--stats_path', type=str, default='data_stats.npz',
                        help='归一化统计量路径（由 train.py 生成）')
    parser.add_argument('--history_path', type=str, default='train_history.npz',
                        help='训练历史路径（由 train.py 生成）')
    parser.add_argument('--data_dir', type=str, default='./EuroSAT_RGB',
                        help='EuroSAT_RGB 数据集目录路径')

    # 模型结构参数（需与训练时一致）
    parser.add_argument('--hidden_dim',   type=int, default=256,
                        help='隐藏层大小（需与训练时一致）')
    parser.add_argument('--activation',   type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='激活函数（需与训练时一致）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 正则化强度（需与训练时一致）')

    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--val_ratio',  type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)

    parser.add_argument('--n_error_samples', type=int, default=9,
                        help='展示的错误样本数量')
    parser.add_argument('--n_weight_vis',    type=int, default=16,
                        help='展示的第一层权重图像数量')

    parser.add_argument('--skip_visualize', action='store_true',
                        help='跳过可视化输出')

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("  EuroSAT_RGB 手工 MLP 分类器 — 测试脚本")
    print("=" * 60)

    # ── 检查模型文件是否存在 ───────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"错误：找不到模型权重文件 {args.model_path}")
        print("请先运行 train.py 训练模型。")
        return

    # ── 1. 加载数据 ────────────────────────────────────────────
    print("\n[步骤 1/3] 加载测试数据...")
    loader = EuroSATDataLoader(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()

    # 加载归一化统计量（确保与训练时一致）
    if os.path.exists(args.stats_path):
        loader.load_stats(args.stats_path)
        print(f"已加载归一化统计量: {args.stats_path}")
    else:
        print(f"警告：未找到 {args.stats_path}，使用当前数据集重新计算统计量")

    # ── 2. 加载模型 ────────────────────────────────────────────
    print(f"\n[步骤 2/3] 加载模型权重: {args.model_path}")
    model = MLP(
        input_dim=X_test.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=10,
        activation=args.activation,
        weight_decay=args.weight_decay,
    )
    model.load_weights(args.model_path)
    print(f"模型: {model}")

    # ── 3. 测试集评估 ──────────────────────────────────────────
    print("\n[步骤 3/3] 在测试集上评估...")
    evaluator = Evaluator(class_names=get_class_names())
    test_acc, conf_mat = evaluator.evaluate(model, X_test, y_test)
    evaluator.print_report(test_acc, conf_mat)

    # ── 4. 可视化（可选）─────────────────────────────────────
    if not args.skip_visualize:
        print("\n[可选] 生成可视化图表...")

        # 4a. 训练曲线（如果存在历史文件）
        if os.path.exists(args.history_path):
            hist_data = np.load(args.history_path)
            history = {
                'train_loss': hist_data['train_loss'].tolist(),
                'val_loss':   hist_data['val_loss'].tolist(),
                'val_acc':    hist_data['val_acc'].tolist(),
                'lr_history': hist_data['lr_history'].tolist(),
            }
            plot_training_curves(history, save_path='training_curves.png')
            print("  训练曲线已保存: training_curves.png")
        else:
            print(f"  跳过训练曲线（未找到 {args.history_path}）")

        # 4b. 混淆矩阵热力图
        plot_confusion_matrix(
            conf_mat,
            class_names=get_class_names(),
            save_path='confusion_matrix.png',
        )
        print("  混淆矩阵热力图已保存: confusion_matrix.png")

        # 4c. 第一层权重可视化
        visualize_weights(
            model,
            n_weights=args.n_weight_vis,
            image_size=64,
            save_path='weight_visualization.png',
        )
        print(f"  权重可视化已保存: weight_visualization.png")

        # 4d. 错例分析
        # 需要原始图像用于可视化（重新加载未归一化的版本）
        errors = evaluator.get_error_samples(
            model, X_test, y_test,
            n=args.n_error_samples,
        )
        visualize_error_samples(
            errors,
            class_names=get_class_names(),
            image_size=64,
            mean=loader.mean,
            std=loader.std,
            save_path='error_analysis.png',
        )
        print(f"  错例分析图已保存: error_analysis.png")

    print("\n测试完成！")
    print(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")


if __name__ == '__main__':
    main()
