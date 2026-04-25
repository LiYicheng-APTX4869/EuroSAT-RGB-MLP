"""
visualize.py — 可视化模块

功能：
1. plot_training_curves: 绘制训练/验证 Loss 曲线和 Accuracy 曲线
2. visualize_weights:    可视化第一层隐藏层权重（reshape 为图像）
3. visualize_error_samples: 展示分类错误的样本
4. plot_confusion_matrix: 绘制混淆矩阵热力图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合无 GUI 环境保存图片
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 支持中文字体（Windows 下使用 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 训练曲线可视化
# ============================================================

def plot_training_curves(history, save_path='training_curves.png', show=False):
    """
    绘制训练过程曲线

    包含三个子图：
    - 左：训练集 Loss 和验证集 Loss（双线）
    - 中：验证集 Accuracy
    - 右：学习率变化曲线

    参数：
        history (dict): Trainer.train() 返回的历史记录
        save_path (str): 图像保存路径
        show (bool): 是否调用 plt.show()（交互式环境使用）
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('训练过程监控', fontsize=14, fontweight='bold')

    # ── 子图1：Loss 曲线 ──────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='训练集 Loss', linewidth=1.5)
    ax.plot(epochs, history['val_loss'],   'r-', label='验证集 Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练集与验证集 Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：Accuracy 曲线 ──────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history['val_acc'], 'g-', label='验证集 Accuracy', linewidth=1.5)
    best_epoch = int(np.argmax(history['val_acc'])) + 1
    best_acc   = max(history['val_acc'])
    ax.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7,
               label=f'最优 Epoch={best_epoch}')
    ax.scatter([best_epoch], [best_acc], color='orange', zorder=5, s=50)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('验证集 Accuracy')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate(f'{best_acc:.3f}',
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 1, best_acc - 0.05),
                fontsize=9, color='orange')

    # ── 子图3：学习率曲线 ─────────────────────────────────────
    ax = axes[2]
    if 'lr_history' in history:
        ax.plot(epochs, history['lr_history'], 'm-', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('学习率衰减曲线')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, '无学习率历史', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {save_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# 2. 第一层权重可视化
# ============================================================

def visualize_weights(model, n_weights=16, image_size=64,
                      save_path='weight_visualization.png', show=False):
    """
    将第一层隐藏层的权重矩阵可视化为图像

    W1 形状: (12288, hidden_dim)
    每列 reshape 为 (64, 64, 3) 的 RGB 图像

    这些"权重图像"类似于卷积滤波器，
    展示了每个神经元对哪种视觉模式最敏感。

    参数：
        model:       MLP 实例（含已训练的 W1）
        n_weights:   展示的权重图像数量
        image_size:  图像尺寸（默认 64）
        save_path:   保存路径
        show:        是否调用 plt.show()
    """
    W1 = model.W1  # (12288, H)
    H = W1.shape[1]
    n = min(n_weights, H)

    # 计算网格大小
    n_cols = min(8, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 1.5, n_rows * 1.5 + 0.5))
    fig.suptitle(f'第一层隐藏层权重可视化（共展示 {n}/{H} 个神经元）',
                 fontsize=12, fontweight='bold')

    # 展平 axes 方便遍历
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]

        if i < n:
            # 取第 i 列权重，reshape 为图像
            w = W1[:, i].reshape(image_size, image_size, 3)

            # 归一化到 [0, 1] 用于显示
            w_min, w_max = w.min(), w.max()
            if w_max - w_min > 1e-8:
                w_disp = (w - w_min) / (w_max - w_min)
            else:
                w_disp = w - w_min

            ax.imshow(w_disp, interpolation='nearest')
            ax.set_title(f'#{i}', fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"权重可视化已保存: {save_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# 3. 错例分析可视化
# ============================================================

def visualize_error_samples(errors, class_names=None, image_size=64,
                              mean=None, std=None,
                              save_path='error_analysis.png', show=False):
    """
    展示分类错误的样本图像

    参数：
        errors (dict): Evaluator.get_error_samples() 的返回值
            - 'X_display': (N, 12288) 图像数据（已归一化）
            - 'y_true':    (N,) 真实标签
            - 'y_pred':    (N,) 预测标签
        class_names (list): 类别名称
        image_size (int):   图像尺寸
        mean, std (ndarray): 归一化统计量（用于反归一化，恢复原始像素）
        save_path:          保存路径
        show:               是否调用 plt.show()
    """
    X_display = errors['X_display']  # 已归一化
    y_true    = errors['y_true']
    y_pred    = errors['y_pred']
    n = len(y_true)

    if n == 0:
        print("没有分类错误的样本！")
        return

    names = class_names if class_names is not None else [str(i) for i in range(10)]

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3, n_rows * 3 + 0.5))
    fig.suptitle('错例分析（分类错误的卫星图像）',
                 fontsize=13, fontweight='bold')

    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]

        if i < n:
            # 反归一化：恢复到 [0,1] 范围的像素值
            img = X_display[i].copy()  # (12288,)
            if mean is not None and std is not None:
                img = img * std + mean   # 反标准化
            # 还原到 [0, 255]（训练时除以255了）
            img = np.clip(img, 0, 1)
            img = img.reshape(image_size, image_size, 3)

            ax.imshow(img, interpolation='nearest')

            true_name = names[y_true[i]] if y_true[i] < len(names) else str(y_true[i])
            pred_name = names[y_pred[i]] if y_pred[i] < len(names) else str(y_pred[i])
            ax.set_title(f'真实: {true_name}\n预测: {pred_name}',
                         fontsize=9,
                         color='red')
        ax.axis('off')

    # 隐藏多余的格子
    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"错例分析图已保存: {save_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# 4. 混淆矩阵热力图
# ============================================================

def plot_confusion_matrix(conf_mat, class_names=None,
                           save_path='confusion_matrix.png',
                           normalize=False, show=False):
    """
    绘制混淆矩阵热力图

    参数：
        conf_mat (ndarray): (C, C) 混淆矩阵
        class_names (list): 类别名称
        save_path (str):    保存路径
        normalize (bool):   是否按行归一化（显示比例）
        show (bool):        是否调用 plt.show()
    """
    n = conf_mat.shape[0]
    names = (class_names if class_names is not None
             else [str(i) for i in range(n)])

    if normalize:
        cm = conf_mat.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / (row_sums + 1e-10)
        fmt = '.2f'
        title = '混淆矩阵（归一化，按行）'
    else:
        cm = conf_mat
        fmt = 'd'
        title = '混淆矩阵（样本数）'

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # 使用短名称
    short_names = [name[:12] for name in names]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('预测类别', fontsize=11)
    ax.set_ylabel('真实类别', fontsize=11)

    # 在格子中标注数值
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text = f'{val:{fmt}}' if fmt == 'd' else f'{val:.2f}'
            ax.text(j, i, text,
                    ha='center', va='center', fontsize=7,
                    color='white' if val > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵热力图已保存: {save_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# 快速测试
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from model import MLP

    print("=== 可视化模块快速测试 ===")

    # 模拟训练历史
    epochs = 50
    history = {
        'train_loss': (2.3 * np.exp(-np.arange(epochs) * 0.05)
                       + np.random.rand(epochs) * 0.05).tolist(),
        'val_loss':   (2.3 * np.exp(-np.arange(epochs) * 0.04)
                       + np.random.rand(epochs) * 0.05).tolist(),
        'val_acc':    (1 - np.exp(-np.arange(epochs) * 0.06)
                       + np.random.rand(epochs) * 0.01).clip(0, 1).tolist(),
        'lr_history': [0.01 * (0.95 ** (e // 10)) for e in range(epochs)],
    }
    plot_training_curves(history, save_path='test_training_curves.png')
    print("训练曲线测试完成")

    # 权重可视化（小模型）
    model = MLP(input_dim=64*64*3, hidden_dim=16, output_dim=10)
    visualize_weights(model, n_weights=8, image_size=64,
                      save_path='test_weight_vis.png')
    print("权重可视化测试完成")

    # 混淆矩阵
    rng = np.random.default_rng(0)
    cm = rng.integers(0, 100, size=(10, 10))
    np.fill_diagonal(cm, rng.integers(200, 500, size=10))
    from data_loader import CLASSES
    plot_confusion_matrix(cm, class_names=CLASSES,
                           save_path='test_confusion_matrix.png')
    print("混淆矩阵测试完成")

    # 错例分析
    rng2 = np.random.default_rng(42)
    errors = {
        'X_display': rng2.random((6, 64*64*3)),
        'y_true': np.array([0, 1, 2, 3, 4, 5]),
        'y_pred': np.array([1, 2, 3, 4, 5, 6]),
    }
    visualize_error_samples(errors, class_names=CLASSES,
                             save_path='test_error_analysis.png')
    print("错例分析测试完成")

    print("\n所有可视化测试通过！")
