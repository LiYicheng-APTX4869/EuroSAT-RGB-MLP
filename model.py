"""
model.py — 三层 MLP 模型（手工实现前向传播与反向传播）

不使用任何深度学习框架（PyTorch/TensorFlow/JAX 等），
仅使用 NumPy 手工实现：
- 前向传播
- 交叉熵损失 + L2 正则化
- 反向传播（解析梯度）
- 模型权重保存/加载
"""

import numpy as np


# ============================================================
# 激活函数
# ============================================================

def relu(z):
    """ReLU 激活函数"""
    return np.maximum(0.0, z)


def relu_grad(z):
    """ReLU 梯度"""
    return (z > 0).astype(np.float64)


def sigmoid(z):
    """Sigmoid 激活函数（数值稳定版本）"""
    # 对正值和负值分别处理，防止溢出
    pos_mask = z >= 0
    result = np.zeros_like(z)
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[~pos_mask])
    result[~pos_mask] = exp_z / (1.0 + exp_z)
    return result


def sigmoid_grad(z):
    """Sigmoid 梯度：σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh_activation(z):
    """Tanh 激活函数"""
    return np.tanh(z)


def tanh_grad(z):
    """Tanh 梯度：1 - tanh²(z)"""
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    'relu':    (relu,             relu_grad),
    'sigmoid': (sigmoid,          sigmoid_grad),
    'tanh':    (tanh_activation,  tanh_grad),
}


# ============================================================
# Softmax（数值稳定版）
# ============================================================

def softmax(z):
    """
    数值稳定的 Softmax

    z: (N, C) 或 (C,)
    返回: 与 z 相同形状的概率矩阵
    """
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


# ============================================================
# 三层 MLP
# ============================================================

class MLP:
    """
    三层全连接神经网络（MLP）

    网络结构：
        输入层 (input_dim=12288)
        → 隐藏层1 (hidden_dim) + 激活
        → 隐藏层2 (hidden_dim) + 激活
        → 输出层 (output_dim=10) → Softmax

    参数：
        input_dim   (int):   输入维度，EuroSAT 为 64*64*3=12288
        hidden_dim  (int):   隐藏层大小（两层相同）
        output_dim  (int):   输出类别数，EuroSAT 为 10
        activation  (str):   激活函数 'relu' | 'sigmoid' | 'tanh'
        weight_decay (float): L2 正则化强度 λ
    """

    def __init__(self, input_dim=12288, hidden_dim=256, output_dim=10,
                 activation='relu', weight_decay=1e-4):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation 必须为 {list(ACTIVATIONS.keys())} 之一")

        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        self.weight_decay = weight_decay
        self.act_fn, self.act_grad = ACTIVATIONS[activation]
        self.activation_name = activation

        # 初始化权重（He 初始化适用于 ReLU；Xavier 适用于 Sigmoid/Tanh）
        self._init_weights()

        # 缓存前向传播中间值（反向传播需要）
        self._cache = {}

    def _init_weights(self):
        """
        权重初始化

        - W: 使用 He 初始化（对 ReLU 友好）
        - b: 初始化为零
        """
        rng = np.random.default_rng(42)
        H = self.hidden_dim
        I = self.input_dim
        O = self.output_dim

        if self.activation_name == 'relu':
            # He 初始化
            scale1 = np.sqrt(2.0 / I)
            scale2 = np.sqrt(2.0 / H)
            scale3 = np.sqrt(2.0 / H)
        else:
            # Xavier 初始化
            scale1 = np.sqrt(1.0 / I)
            scale2 = np.sqrt(1.0 / H)
            scale3 = np.sqrt(1.0 / H)

        self.W1 = rng.standard_normal((I, H)) * scale1  # (12288, H)
        self.b1 = np.zeros(H)                            # (H,)
        self.W2 = rng.standard_normal((H, H)) * scale2  # (H, H)
        self.b2 = np.zeros(H)                            # (H,)
        self.W3 = rng.standard_normal((H, O)) * scale3  # (H, 10)
        self.b3 = np.zeros(O)                            # (10,)

    def forward(self, X):
        """
        前向传播

        参数：
            X: (N, input_dim) — 输入样本（已归一化）

        返回：
            probs: (N, output_dim) — Softmax 概率分布

        同时将中间值缓存到 self._cache，供 backward() 使用。
        """
        # 第一层
        z1 = X @ self.W1 + self.b1           # (N, H)
        a1 = self.act_fn(z1)                  # (N, H)

        # 第二层
        z2 = a1 @ self.W2 + self.b2          # (N, H)
        a2 = self.act_fn(z2)                  # (N, H)

        # 输出层
        z3 = a2 @ self.W3 + self.b3          # (N, 10)
        probs = softmax(z3)                   # (N, 10)

        # 缓存中间值
        self._cache = {
            'X': X, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'probs': probs
        }

        return probs

    def compute_loss(self, probs, y):
        """
        计算交叉熵损失 + L2 正则化

        参数：
            probs: (N, 10) — Softmax 概率
            y:     (N,)   — 真实标签 [0, 9]

        返回：
            loss (float): 标量损失值
        """
        N = len(y)
        # 数值稳定的 log（防止 log(0)）
        log_probs = np.log(probs[np.arange(N), y] + 1e-12)
        cross_entropy = -np.mean(log_probs)

        # L2 正则化（只对权重矩阵，不对偏置）
        l2_reg = (self.weight_decay / 2.0) * (
            np.sum(self.W1 ** 2) +
            np.sum(self.W2 ** 2) +
            np.sum(self.W3 ** 2)
        )

        return cross_entropy + l2_reg

    def backward(self, probs, y):
        """
        反向传播，计算所有参数的梯度

        参数：
            probs: (N, 10) — 前向传播输出的概率
            y:     (N,)   — 真实标签

        返回：
            grads (dict): 包含 'W1','b1','W2','b2','W3','b3' 的梯度字典

        手工推导的梯度：
        ┌─────────────────────────────────────────────────────────┐
        │ Softmax + CrossEntropy 合并梯度：                         │
        │   dL/dz3 = (probs - one_hot(y)) / N                    │
        │                                                          │
        │ 输出层：                                                  │
        │   dL/dW3 = a2.T @ dz3 + λ*W3                           │
        │   dL/db3 = sum(dz3, axis=0)                             │
        │                                                          │
        │ 隐藏层2（反传激活梯度）：                                   │
        │   dL/da2 = dz3 @ W3.T                                   │
        │   dL/dz2 = da2 * act_grad(z2)                           │
        │   dL/dW2 = a1.T @ dz2 + λ*W2                           │
        │   dL/db2 = sum(dz2, axis=0)                             │
        │                                                          │
        │ 隐藏层1（反传激活梯度）：                                   │
        │   dL/da1 = dz2 @ W2.T                                   │
        │   dL/dz1 = da1 * act_grad(z1)                           │
        │   dL/dW1 = X.T @ dz1 + λ*W1                            │
        │   dL/db1 = sum(dz1, axis=0)                             │
        └─────────────────────────────────────────────────────────┘
        """
        N = len(y)
        cache = self._cache
        X  = cache['X']
        z1 = cache['z1']
        a1 = cache['a1']
        z2 = cache['z2']
        a2 = cache['a2']

        # ── 输出层梯度 ────────────────────────────────────────────
        # Softmax + Cross-Entropy 合并梯度
        dz3 = probs.copy()
        dz3[np.arange(N), y] -= 1.0
        dz3 /= N                                # (N, 10)

        dW3 = a2.T @ dz3 + self.weight_decay * self.W3   # (H, 10)
        db3 = dz3.sum(axis=0)                             # (10,)

        # ── 隐藏层2梯度 ───────────────────────────────────────────
        da2 = dz3 @ self.W3.T                             # (N, H)
        dz2 = da2 * self.act_grad(z2)                     # (N, H)

        dW2 = a1.T @ dz2 + self.weight_decay * self.W2   # (H, H)
        db2 = dz2.sum(axis=0)                             # (H,)

        # ── 隐藏层1梯度 ───────────────────────────────────────────
        da1 = dz2 @ self.W2.T                             # (N, H)
        dz1 = da1 * self.act_grad(z1)                     # (N, H)

        dW1 = X.T @ dz1 + self.weight_decay * self.W1    # (I, H)
        db1 = dz1.sum(axis=0)                             # (H,)

        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3,
        }

    def predict(self, X):
        """
        预测类别

        参数：
            X: (N, input_dim)

        返回：
            preds: (N,) — 预测类别索引
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def save_weights(self, path):
        """保存模型权重到 .npz 文件"""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 hidden_dim=np.array(self.hidden_dim),
                 activation=np.array(self.activation_name),
                 weight_decay=np.array(self.weight_decay))
        print(f"模型权重已保存至 {path}")

    def load_weights(self, path):
        """从 .npz 文件加载模型权重"""
        data = np.load(path, allow_pickle=True)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        print(f"模型权重已从 {path} 加载")

    def get_params(self):
        """返回所有参数（用于超参数搜索时重置）"""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
            'W3': self.W3.copy(), 'b3': self.b3.copy(),
        }

    def set_params(self, params):
        """设置所有参数"""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
        self.W3 = params['W3'].copy()
        self.b3 = params['b3'].copy()

    def __repr__(self):
        return (f"MLP(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                f"output_dim={self.output_dim}, activation='{self.activation_name}', "
                f"weight_decay={self.weight_decay})")


# ============================================================
# 数值梯度检验（用于验证反向传播正确性）
# ============================================================

def numerical_gradient_check(model, X, y, eps=1e-5):
    """
    用有限差分法验证反向传播梯度的正确性

    计算方法：
        grad ≈ (L(w+eps) - L(w-eps)) / (2*eps)

    相对误差应 < 1e-5 才算通过

    参数：
        model: MLP 实例
        X:     小批量输入，建议取 3-5 个样本
        y:     对应标签
        eps:   有限差分步长

    返回：
        max_rel_error (float): 最大相对误差
    """
    # 计算解析梯度
    probs = model.forward(X)
    loss = model.compute_loss(probs, y)
    analytic_grads = model.backward(probs, y)

    max_rel_error = 0.0

    for param_name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        param = getattr(model, param_name)
        analytic = analytic_grads[param_name]

        # 只检查前 min(10, size) 个参数，避免太慢
        indices = np.ndindex(*param.shape)
        checked = 0

        for idx in indices:
            if checked >= 20:
                break
            # 正向扰动
            param[idx] += eps
            probs_p = model.forward(X)
            loss_p = model.compute_loss(probs_p, y)

            # 负向扰动
            param[idx] -= 2 * eps
            probs_m = model.forward(X)
            loss_m = model.compute_loss(probs_m, y)

            # 恢复
            param[idx] += eps

            numeric_grad = (loss_p - loss_m) / (2 * eps)
            analytic_grad = analytic[idx]

            denom = max(abs(numeric_grad), abs(analytic_grad)) + 1e-10
            rel_error = abs(numeric_grad - analytic_grad) / denom
            max_rel_error = max(max_rel_error, rel_error)
            checked += 1

    return max_rel_error


if __name__ == '__main__':
    print("=== 测试 MLP 前向/反向传播 ===")
    # 小规模测试
    N, I, H, O = 5, 20, 16, 10
    rng = np.random.default_rng(0)
    X_test = rng.standard_normal((N, I))
    y_test = rng.integers(0, O, size=N)

    for act in ['relu', 'sigmoid', 'tanh']:
        model = MLP(input_dim=I, hidden_dim=H, output_dim=O,
                    activation=act, weight_decay=1e-3)
        probs = model.forward(X_test)
        loss = model.compute_loss(probs, y_test)
        grads = model.backward(probs, y_test)

        # 梯度检验
        max_err = numerical_gradient_check(model, X_test, y_test)
        status = "✓ PASS" if max_err < 1e-5 else f"✗ FAIL (err={max_err:.2e})"
        print(f"  activation={act:8s}, loss={loss:.4f}, grad_check: {status}")

    print("\n=== 权重保存/加载测试 ===")
    import tempfile, os
    model = MLP(input_dim=I, hidden_dim=H, output_dim=O)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmp_path = f.name
    model.save_weights(tmp_path)
    model2 = MLP(input_dim=I, hidden_dim=H, output_dim=O)
    model2.load_weights(tmp_path)
    os.unlink(tmp_path)
    assert np.allclose(model.W1, model2.W1), "权重加载失败！"
    print("  权重保存/加载测试: ✓ PASS")
