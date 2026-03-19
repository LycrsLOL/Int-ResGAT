# Int-ResGAT

**Int-ResGAT** 是一个基于图神经网络 (GNN) 的酶功能注释（多标签 EC 编号分类）模型。该项目通过整合 EvolutionaryScale 的 ESM3 序列嵌入与蛋白质 3D 空间结构图，实现对蛋白质酶功能的高精度预测。

## 核心特性

- **序列与结构整合**：自动下载/补全 PDB 结构，并结合 ESM3 提取的高维序列特征 (1536-dim)，构建基于空间距离（如 <= 10.0Å）的残基相互作用图。
- **改进的图网络架构**：内置增强版的 `HE_ResGATConv` 模型，结合了 GATv2Conv、残差连接 (Residuals)、跳跃知识连接 (Jumping Knowledge, JK) 以及 Mean+Max 双重池化机制。
- **稳健的模型训练**：
  - 采用 **Focal Loss** 和特定的偏置初始化策略（如 `-log((1-pi)/pi)`），有效应对长尾分布和极度类别不平衡（预测超过 1800 种 EC 分类）。
  - 支持延迟加载 (Lazy Loading) 以降低大规模图数据的内存占用。
- **分布式计算 (DDP)**：原生支持 PyTorch Distributed Data Parallel，包含健壮的多进程上下文管理及异常处理，适配多 GPU 训练环境。

## 目录结构

```text
Int_ResGAT/
├── dataloader/      # 数据处理模块：PDB 下载、结构清洗补全、ESM3 特征提取与图构建
├── train/           # 训练模块：模型定义 (models.py)、损失函数、指标计算与训练循环
├── main.py          # 项目主入口，支持 `initialize` 和 `train` 两种运行模式
├── config.yaml      # 核心配置文件，管理超参数、路径及设备信息
├── dist_train.sh    # 多 GPU 分布式训练 (torchrun) 启动脚本
└── requirements.txt # 项目依赖
```

## 快速开始

### 1. 环境准备

推荐使用 Conda 或虚拟环境：

```bash
pip install -r requirements.txt
```

*注意：ESM3 的权重文件及代码库需预先配置（见 `config.yaml` 和 `dataloader.py` 中的路径配置）。*

### 2. 数据集准备

请确保 `config.yaml` 中指定的 `database_path`（如 `./data/database.csv`）包含必要的列，如 `uniprot_id`、`pdb_id`、`ec_numbers` 等。

### 3. 数据初始化 (Initialize Mode)

此阶段将自动完成结构下载、利用 ESM3 模型提取节点特征、并构建用于训练的 PyTorch Geometric 图数据：

```bash
python main.py --mode initialize --config config.yaml
```

### 4. 模型训练 (Train Mode)

- **单卡训练**：
  ```bash
  python main.py --mode train --config config.yaml
  ```
- **多卡分布式训练 (DDP)**：
  推荐使用提供的 shell 脚本启动，脚本会自动检测可用的 GPU 数量并配置环境变量：
  ```bash
  bash dist_train.sh
  ```

## 配置说明 (`config.yaml`)

你可以直接在 `config.yaml` 中调整以下核心参数：
- `initialize.max_edge_distance`: 建图时的最大残基距离阈值（默认 10.0）。
- `train.batch_size` / `train.learning_rate`: 训练批次与学习率。
- `train.focal_loss_gamma`: Focal Loss 的伽马值，用于难易样本挖掘。
- `train.hidden_dim` / `train.num_layers` / `train.heads`: GNN 结构超参数。
