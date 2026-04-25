"""
trainer.py — 训练循环模块

实现：
- Mini-batch SGD 训练
- 学习率衰减（Step Decay）
- 每 epoch 评估验证集，自动保存最优权重
- 记录 train_loss, val_loss, val_acc 历史曲线
"""

import numpy as np
from tqdm import tqdm


class Trainer:
    """
    MLP 训练器

    参数：
        model:       MLP 实例
        lr:          初始学习率
        lr_decay:    学习率衰减因子（每 decay_step 个 epoch 乘以该值）
        decay_step:  学习率衰减间隔（epoch 数）
        batch_size:  Mini-batch 大小
        epochs:      最大训练轮数
        save_path:   最优模型权重保存路径（None 则不保存）
        verbose:     是否打印训练进度
    """

    def __init__(self, model, lr=0.01, lr_decay=0.95, decay_step=10,
                 batch_size=64, epochs=100, save_path='best_model.npz',
                 verbose=True):
        self.model      = model
        self.lr         = lr
        self.lr_decay   = lr_decay
        self.decay_step = decay_step
        self.batch_size = batch_size
        self.epochs     = epochs
        self.save_path  = save_path
        self.verbose    = verbose

        # 当前学习率（训练过程中会变化）
        self.current_lr = lr

    def train(self, X_train, y_train, X_val, y_val):
        """
        完整训练流程

        参数：
            X_train: (N_train, D) — 训练集特征
            y_train: (N_train,)   — 训练集标签
            X_val:   (N_val, D)   — 验证集特征
            y_val:   (N_val,)     — 验证集标签

        返回：
            history (dict): {
                'train_loss': [...],  # 每个 epoch 的训练损失
                'val_loss':   [...],  # 每个 epoch 的验证损失
                'val_acc':    [...],  # 每个 epoch 的验证准确率
                'lr_history': [...],  # 每个 epoch 的学习率
            }
        """
        N = len(y_train)
        history = {
            'train_loss': [],
            'val_loss':   [],
            'val_acc':    [],
            'lr_history': [],
        }
        best_val_acc = -1.0
        self.current_lr = self.lr

        for epoch in range(1, self.epochs + 1):
            # ── 学习率衰减 ────────────────────────────────────────
            self.current_lr = self._lr_schedule(epoch)

            # ── Mini-batch SGD ────────────────────────────────────
            # 随机打乱训练集
            perm = np.random.permutation(N)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 前向传播
                probs = self.model.forward(X_batch)
                batch_loss = self.model.compute_loss(probs, y_batch)

                # 反向传播
                grads = self.model.backward(probs, y_batch)

                # 参数更新（SGD）
                self._update_params(grads)

                epoch_loss += batch_loss
                n_batches  += 1

            train_loss = epoch_loss / n_batches

            # ── 验证集评估 ────────────────────────────────────────
            val_loss, val_acc = self._evaluate(X_val, y_val)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr_history'].append(self.current_lr)

            # 保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.save_path is not None:
                    self.model.save_weights(self.save_path)

            # 打印进度
            if self.verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"lr={self.current_lr:.6f} | "
                      f"train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_acc={val_acc:.4f} | "
                      f"best_val_acc={best_val_acc:.4f}")

        if self.verbose:
            print(f"\n训练完成！最优验证集准确率: {best_val_acc:.4f}")

        return history

    def _update_params(self, grads, clip_norm=5.0):
        """
        SGD 参数更新：W = W - lr * dL/dW

        clip_norm: 梯度裁剪阈值（全局 L2 范数），防止梯度爆炸
        """
        # 计算全局梯度 L2 范数
        total_norm = 0.0
        for g in grads.values():
            total_norm += np.sum(g ** 2)
        total_norm = np.sqrt(total_norm)

        # 如果范数超过阈值，按比例缩小所有梯度
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}

        self.model.W1 -= self.current_lr * grads['W1']
        self.model.b1 -= self.current_lr * grads['b1']
        self.model.W2 -= self.current_lr * grads['W2']
        self.model.b2 -= self.current_lr * grads['b2']
        self.model.W3 -= self.current_lr * grads['W3']
        self.model.b3 -= self.current_lr * grads['b3']

    def _lr_schedule(self, epoch):
        """
        Step Decay 学习率衰减策略

        每 decay_step 个 epoch，学习率乘以 lr_decay
        lr(t) = lr_init * lr_decay ^ floor(t / decay_step)
        """
        n_decays = (epoch - 1) // self.decay_step
        return self.lr * (self.lr_decay ** n_decays)

    def _evaluate(self, X, y):
        """
        在给定数据集上计算损失和准确率

        参数：
            X: (N, D)
            y: (N,)

        返回：
            loss (float): 平均损失
            acc  (float): 分类准确率 [0, 1]
        """
        # 分批处理避免内存溢出（验证集较大时）
        batch_size = 512
        N = len(y)
        losses = []
        preds  = []

        for start in range(0, N, batch_size):
            end = start + batch_size
            X_b = X[start:end]
            y_b = y[start:end]

            probs = self.model.forward(X_b)
            loss  = self.model.compute_loss(probs, y_b)
            pred  = np.argmax(probs, axis=1)

            losses.append(loss * len(y_b))
            preds.append(pred)

        avg_loss = sum(losses) / N
        all_preds = np.concatenate(preds)
        acc = np.mean(all_preds == y)

        return avg_loss, acc


if __name__ == '__main__':
    # 快速验证训练器可运行
    import sys
    sys.path.insert(0, '.')
    from model import MLP

    print("=== 训练器快速测试（小规模随机数据）===")
    rng = np.random.default_rng(0)
    N_train, N_val, D, C = 500, 100, 64, 10

    X_train = rng.standard_normal((N_train, D))
    y_train = rng.integers(0, C, size=N_train)
    X_val   = rng.standard_normal((N_val, D))
    y_val   = rng.integers(0, C, size=N_val)

    model   = MLP(input_dim=D, hidden_dim=32, output_dim=C,
                  activation='relu', weight_decay=1e-4)
    trainer = Trainer(model, lr=0.05, lr_decay=0.9, decay_step=5,
                      batch_size=32, epochs=30, save_path=None, verbose=True)

    history = trainer.train(X_train, y_train, X_val, y_val)
    print(f"\n最终 val_acc: {history['val_acc'][-1]:.4f}")
    print(f"train_loss 前后: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    print("测试通过！")
