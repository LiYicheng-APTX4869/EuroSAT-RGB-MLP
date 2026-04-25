# EuroSAT RGB 遥感图像分类 —— 手工实现 MLP

基于 **NumPy 手工实现**的三层多层感知机（MLP），对 EuroSAT\_RGB 卫星遥感图像进行 10 类地物分类，不依赖任何深度学习框架（PyTorch / TensorFlow）。

- **测试集准确率**：66.27%
- **数据集**：EuroSAT\_RGB，27,000 张图像，10 类，64×64 RGB

---

## 项目结构

```
hw1/
├── data_loader.py      # 数据集加载与预处理
├── model.py            # MLP 模型（前向传播、反向传播、权重初始化）
├── trainer.py          # Mini-batch SGD 训练循环
├── evaluator.py        # 准确率、混淆矩阵、错例收集
├── visualize.py        # 训练曲线、权重可视化、错例分析
├── hparam_search.py    # 超参数网格搜索
├── train.py            # 主训练脚本
└── test.py             # 测试评估脚本
```

---

## 环境依赖

Python 3.8 及以上，安装以下依赖包：

```bash
pip install numpy pillow matplotlib tqdm
```

或使用 conda 创建虚拟环境：

```bash
conda create -n hw1_eurosat python=3.9
conda activate hw1_eurosat
pip install numpy pillow matplotlib tqdm
```

---

## 数据集准备

下载 [EuroSAT\_RGB](https://github.com/phelber/EuroSAT) 数据集，解压后放置于项目根目录：

```
hw1/
└── EuroSAT_RGB/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

---

## 训练

**直接使用最优超参数训练（推荐）：**

```bash
python train.py --hidden_dim 512 --lr 0.01 --weight_decay 0.001 --activation relu --epochs 100
```

**先进行超参数网格搜索，再训练（约需 5 小时）：**

```bash
python train.py --run_hparam_search --search_epochs 20
```

**主要训练参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | `./EuroSAT_RGB` | 数据集路径 |
| `--hidden_dim` | 256 | 隐藏层神经元数量 |
| `--activation` | relu | 激活函数（relu / tanh / sigmoid） |
| `--lr` | 0.01 | 初始学习率 |
| `--lr_decay` | 0.95 | Step Decay 衰减因子 |
| `--decay_step` | 10 | 每隔多少 epoch 衰减一次 |
| `--weight_decay` | 1e-4 | L2 正则化强度 |
| `--batch_size` | 64 | Mini-batch 大小 |
| `--epochs` | 100 | 训练轮数 |
| `--run_hparam_search` | False | 是否先进行超参数搜索 |
| `--search_epochs` | 20 | 搜索时每组训练的轮数 |

训练完成后将生成以下文件：

- `best_model.npz` —— 验证集最优模型权重
- `train_history.npz` —— 训练历史（Loss / Accuracy）
- `training_curves.png` —— 训练过程可视化图
- `data_stats.npz` —— 归一化统计量

---

## 测试

```bash
python test.py
```

自动加载 `best_model.npz` 和 `data_stats.npz`，在测试集上评估并生成：

- 各类别 Precision / Recall / F1 报告
- 混淆矩阵热力图（`confusion_matrix.png`）
- 第一层权重可视化（`weight_visualization.png`）
- 错例分析图（`error_analysis.png`）

---

## 预训练模型下载

已训练好的模型权重可从 Google Drive 下载：

> **[best_model.npz — Google Drive](https://drive.google.com/file/d/1yspa69EV_nvSsEo8WaOathvE_De8ZsYA/view?usp=drive_link)**

下载后放置于项目根目录，直接运行 `python test.py` 即可复现结果。

---

## 实验结果

| 指标 | 数值 |
|---|---|
| 测试集准确率 | **66.27%** |
| 宏平均 Precision | 65.56% |
| 宏平均 Recall | 65.94% |
| 宏平均 F1 | 65.64% |

各类别 F1 分数：

| 类别 | F1 |
|---|---|
| Forest（森林） | **0.8688** |
| SeaLake（海洋/湖泊） | 0.8532 |
| Industrial（工业区） | 0.7645 |
| Pasture（牧场） | 0.7278 |
| AnnualCrop（一年生作物） | 0.6784 |
| Residential（居住区） | 0.6272 |
| River（河流） | 0.5984 |
| HerbaceousVegetation（草本植被） | 0.5586 |
| PermanentCrop（多年生作物） | 0.4636 |
| Highway（公路） | 0.4229 |

---

## 最优超参数

通过 54 组网格搜索确定：

```
lr            = 0.01
hidden_dim    = 512
weight_decay  = 0.001
activation    = relu
batch_size    = 64
epochs        = 100
lr_decay      = 0.95（每 10 个 epoch 衰减一次）
```
