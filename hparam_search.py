"""
hparam_search.py — 超参数搜索模块

实现网格搜索（Grid Search）：
- 在给定的超参数组合空间中训练模型
- 记录每组超参数的验证集准确率
- 输出排序后的结果表格
- 支持保存/加载搜索结果
"""

import numpy as np
import itertools
import time
import json
import os


def grid_search(X_train, y_train, X_val, y_val,
                search_space=None,
                fixed_params=None,
                epochs=20,
                batch_size=64,
                save_path='hparam_results.json',
                verbose=True):
    """
    网格超参数搜索

    在 search_space 定义的所有参数组合上训练模型，
    选择验证集准确率最高的一组作为最佳超参数。

    参数：
        X_train, y_train: 训练集
        X_val, y_val:     验证集
        search_space (dict): 超参数搜索空间，例如：
            {
                'lr':          [0.1, 0.01, 0.001],
                'hidden_dim':  [128, 256, 512],
                'weight_decay':[0, 1e-4, 1e-3],
                'activation':  ['relu', 'tanh'],
            }
        fixed_params (dict): 固定不变的超参数
        epochs (int):       每组超参数的训练轮数（搜索阶段少训练）
        batch_size (int):   批大小
        save_path (str):    结果保存路径
        verbose (bool):     是否打印进度

    返回：
        results (list of dict): 每组超参数的结果，包含 val_acc
            结果按 val_acc 降序排列
        best_params (dict): 最优超参数
    """
    # 延迟导入，避免循环依赖
    from model import MLP
    from trainer import Trainer

    if search_space is None:
        search_space = {
            'lr':           [0.05, 0.01, 0.001],
            'hidden_dim':   [128, 256, 512],
            'weight_decay': [0.0, 1e-4, 1e-3],
            'activation':   ['relu', 'tanh'],
        }

    if fixed_params is None:
        fixed_params = {}

    # 生成所有参数组合
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(itertools.product(*param_values))
    n_total = len(all_combinations)

    if verbose:
        print(f"超参数搜索：共 {n_total} 种组合，每组训练 {epochs} 个 epoch")
        print("搜索空间：")
        for k, v in search_space.items():
            print(f"  {k}: {v}")
        print()

    results = []
    start_time = time.time()

    for idx, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))
        params.update(fixed_params)

        if verbose:
            print(f"[{idx+1}/{n_total}] 参数: {params}", end=' ... ')

        # 创建模型
        model = MLP(
            input_dim=X_train.shape[1],
            hidden_dim=params.get('hidden_dim', 256),
            output_dim=len(np.unique(y_train)),
            activation=params.get('activation', 'relu'),
            weight_decay=params.get('weight_decay', 1e-4),
        )

        # 训练（不保存权重，只关注 val_acc）
        trainer = Trainer(
            model=model,
            lr=params.get('lr', 0.01),
            lr_decay=params.get('lr_decay', 0.95),
            decay_step=params.get('decay_step', 10),
            batch_size=batch_size,
            epochs=epochs,
            save_path=None,
            verbose=False,
        )

        history = trainer.train(X_train, y_train, X_val, y_val)

        # 取最优 val_acc（而非最后一个 epoch）
        best_val_acc = max(history['val_acc'])
        final_val_acc = history['val_acc'][-1]

        result = {
            **params,
            'best_val_acc':  best_val_acc,
            'final_val_acc': final_val_acc,
        }
        results.append(result)

        if verbose:
            print(f"best_val_acc={best_val_acc:.4f}")

    # 按 best_val_acc 降序排列
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n搜索完成！用时 {elapsed:.1f} 秒")
        print_search_results(results, top_n=10)

    # 保存结果
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"搜索结果已保存至 {save_path}")

    best_params = {k: v for k, v in results[0].items()
                   if k not in ('best_val_acc', 'final_val_acc')}

    return results, best_params


def print_search_results(results, top_n=10):
    """打印 Top-N 超参数搜索结果"""
    print(f"\nTop {min(top_n, len(results))} 超参数组合（按验证集准确率排序）：")
    print("-" * 80)

    # 获取所有超参数列名
    skip_keys = {'best_val_acc', 'final_val_acc'}
    param_keys = [k for k in results[0].keys() if k not in skip_keys]

    # 表头
    header = "排名  " + "  ".join(f"{k:>12}" for k in param_keys)
    header += f"  {'best_val_acc':>14}  {'final_val_acc':>14}"
    print(header)
    print("-" * 80)

    for rank, r in enumerate(results[:top_n], 1):
        row = f"{rank:3d}   "
        row += "  ".join(f"{str(r.get(k, 'N/A')):>12}" for k in param_keys)
        row += f"  {r['best_val_acc']:>14.4f}  {r['final_val_acc']:>14.4f}"
        print(row)

    print()


def load_search_results(path):
    """加载超参数搜索结果"""
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    print("=== 超参数搜索快速测试（小规模数据）===")
    rng = np.random.default_rng(0)
    N_train, N_val, D, C = 300, 100, 32, 5

    X_train = rng.standard_normal((N_train, D))
    y_train = rng.integers(0, C, size=N_train)
    X_val   = rng.standard_normal((N_val, D))
    y_val   = rng.integers(0, C, size=N_val)

    small_space = {
        'lr':           [0.1, 0.01],
        'hidden_dim':   [16, 32],
        'weight_decay': [0, 1e-4],
        'activation':   ['relu'],
    }

    results, best_params = grid_search(
        X_train, y_train, X_val, y_val,
        search_space=small_space,
        epochs=10,
        batch_size=32,
        save_path=None,
        verbose=True,
    )

    print(f"\n最优超参数: {best_params}")
    print("测试通过！")
