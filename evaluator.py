"""
evaluator.py — 测试评估模块

实现：
- 分类准确率（Accuracy）计算
- 混淆矩阵（10×10）打印和返回
- 错误分类样本收集（用于错例分析）
"""

import numpy as np


class Evaluator:
    """
    模型评估器

    用于在测试集上评估 MLP 的分类性能，
    输出 Accuracy 和混淆矩阵，并收集分类错误的样本。
    """

    def __init__(self, class_names=None):
        """
        参数：
            class_names (list): 类别名称列表（用于打印混淆矩阵标签）
        """
        self.class_names = class_names

    def evaluate(self, model, X, y, batch_size=512):
        """
        在数据集上计算 Accuracy 和混淆矩阵

        参数：
            model:      MLP 实例
            X:          (N, D) — 输入特征
            y:          (N,)   — 真实标签
            batch_size: 分批推理时的批大小

        返回：
            accuracy (float): 分类准确率 [0, 1]
            conf_mat (ndarray): (C, C) 混淆矩阵
                conf_mat[i][j] = 真实类别 i 被预测为类别 j 的数量
        """
        preds = self._predict_batched(model, X, batch_size)
        accuracy = np.mean(preds == y)
        conf_mat = self._compute_confusion_matrix(y, preds)
        return accuracy, conf_mat

    def _predict_batched(self, model, X, batch_size=512):
        """分批推理，避免大数据集内存溢出"""
        N = len(X)
        preds = []
        for start in range(0, N, batch_size):
            end = start + batch_size
            probs = model.forward(X[start:end])
            preds.append(np.argmax(probs, axis=1))
        return np.concatenate(preds)

    def _compute_confusion_matrix(self, y_true, y_pred):
        """
        计算混淆矩阵

        conf_mat[i][j] 表示：真实类别为 i，预测为 j 的样本数
        """
        n_classes = max(y_true.max(), y_pred.max()) + 1
        conf_mat = np.zeros((n_classes, n_classes), dtype=np.int64)
        for true, pred in zip(y_true, y_pred):
            conf_mat[true][pred] += 1
        return conf_mat

    def print_report(self, accuracy, conf_mat):
        """
        打印格式化的评估报告

        包括：
        - 整体 Accuracy
        - 各类别的 Precision, Recall, F1
        - 混淆矩阵
        """
        n_classes = conf_mat.shape[0]
        names = (self.class_names if self.class_names is not None
                 else [str(i) for i in range(n_classes)])

        print("=" * 70)
        print(f"  整体准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("=" * 70)

        # 各类别指标
        print(f"\n{'类别':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'支持数':>8}")
        print("-" * 70)

        precisions, recalls, f1s = [], [], []
        for i in range(n_classes):
            tp = conf_mat[i, i]
            fp = conf_mat[:, i].sum() - tp
            fn = conf_mat[i, :].sum() - tp

            precision = tp / (tp + fp + 1e-10)
            recall    = tp / (tp + fn + 1e-10)
            f1        = 2 * precision * recall / (precision + recall + 1e-10)
            support   = conf_mat[i, :].sum()

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            name = names[i] if i < len(names) else str(i)
            print(f"  {name:<23} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>8d}")

        print("-" * 70)
        print(f"  {'宏平均':<23} {np.mean(precisions):>10.4f} "
              f"{np.mean(recalls):>10.4f} {np.mean(f1s):>10.4f}")

        # 混淆矩阵
        print("\n混淆矩阵（行=真实类别，列=预测类别）：")
        # 截短类别名（最多12字符）
        short_names = [n[:10] for n in names]
        header = " " * 22 + "".join(f"{n:>8}" for n in short_names)
        print(header)
        print("-" * (22 + 8 * n_classes))
        for i in range(n_classes):
            row_name = names[i][:20]
            row = conf_mat[i]
            print(f"  {row_name:<20} " +
                  "".join(f"{v:>8d}" for v in row))
        print()

    def get_error_samples(self, model, X, y, X_raw=None,
                          n=9, batch_size=512):
        """
        获取分类错误的样本

        参数：
            model:   MLP 实例
            X:       (N, D) — 归一化后的特征
            y:       (N,)   — 真实标签
            X_raw:   (N, D) — 原始未归一化特征（可选，用于可视化）
            n:       最多返回的错误样本数量
            batch_size: 批大小

        返回：
            error_info (dict): {
                'indices':      错误样本在数据集中的索引
                'y_true':       真实标签
                'y_pred':       预测标签
                'X_display':    用于显示的图像数据（X_raw 或 X）
            }
        """
        preds = self._predict_batched(model, X, batch_size)
        error_mask = preds != y
        error_indices = np.where(error_mask)[0]

        # 随机抽取 n 个错误样本
        if len(error_indices) > n:
            np.random.seed(42)
            chosen = np.random.choice(error_indices, size=n, replace=False)
        else:
            chosen = error_indices

        display = X_raw[chosen] if X_raw is not None else X[chosen]

        return {
            'indices': chosen,
            'y_true':  y[chosen],
            'y_pred':  preds[chosen],
            'X_display': display,
        }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from model import MLP

    print("=== 评估器快速测试 ===")
    rng = np.random.default_rng(0)
    N, D, C = 200, 64, 10

    X = rng.standard_normal((N, D))
    y = rng.integers(0, C, size=N)

    model = MLP(input_dim=D, hidden_dim=32, output_dim=C)
    evaluator = Evaluator(class_names=[f'class{i}' for i in range(C)])

    acc, cm = evaluator.evaluate(model, X, y)
    print(f"测试准确率: {acc:.4f}")
    evaluator.print_report(acc, cm)

    errors = evaluator.get_error_samples(model, X, y, n=5)
    print(f"错误样本数量: {len(errors['indices'])}")
    print(f"真实标签: {errors['y_true']}")
    print(f"预测标签: {errors['y_pred']}")
    print("测试通过！")
