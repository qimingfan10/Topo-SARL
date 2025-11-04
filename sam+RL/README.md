# SAM2 + 强化学习：血管狭窄分割

## 项目简介

本项目实现了**阶段 A：最小可行原型（候选选择器）**，结合 SAM2（Segment Anything Model 2）与 PPO（Proximal Policy Optimization）强化学习算法，用于血管狭窄分割任务。

### 核心思路

- **SAM2**：生成多个候选掩膜（通过不同的提示点和阈值）
- **PPO RL**：学习如何从候选中选择/融合，以最大化分割质量
- **奖励函数**：IoU、clDice（中心线 Dice）、拓扑约束、vesselness 等

---

## 目录结构

```
sam+RL/
├── config/
│   └── default.yaml          # 默认配置文件
├── models/
│   ├── sam2_wrapper.py       # SAM2 包装器（多候选生成）
│   └── ppo_policy.py         # PPO 策略网络
├── env/
│   └── candidate_selection_env.py  # RL 环境（候选选择）
├── rewards/
│   └── reward_functions.py   # 奖励函数（IoU/clDice/vesselness等）
├── utils/
│   └── data_loader.py        # 数据加载工具
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── requirements.txt          # 依赖包列表
└── README.md                 # 本文档
```

---

## 环境配置

### 1. 依赖安装

**重要**：本项目依赖 SAM2，需要先安装 SAM2：

```bash
cd /home/ubuntu/sam2
pip install -e .
```

然后安装本项目的依赖：

```bash
cd /home/ubuntu/sam+RL
pip install -r requirements.txt
```

### 2. 检查 CUDA

确保 PyTorch 支持 CUDA（SAM2 推荐 `torch>=2.5.1`）：

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 下载 SAM2 权重

SAM2.1 Large 权重已位于：`/home/ubuntu/sam2.1_hiera_large.pt`

如需其他版本，请参考 [SAM2 官方文档](https://github.com/facebookresearch/sam2)。

---

## 配置文件

配置文件位于 `config/default.yaml`，主要配置项：

- **SAM2 配置**：权重路径、模型配置、设备、半精度
- **候选生成配置**：初始点数、阈值列表、多候选模式
- **RL 环境配置**：最大步数、图像尺寸、观察特征维度
- **奖励函数配置**：各项权重（IoU、clDice、拓扑、成本等）
- **PPO 训练配置**：学习率、批次大小、gamma、熵系数等
- **数据配置**：训练/验证图像和掩膜目录

**重要**：根据你的数据集路径修改 `data.train_image_dir` 和 `data.train_mask_dir`（如果有标注）。

---

## 训练

### 基本训练命令

```bash
cd /home/ubuntu/sam+RL

# 使用默认配置训练
python train.py

# 或指定配置文件
python train.py --config config/default.yaml
```

### 训练过程

1. **数据加载**：从 `train_image_dir` 加载图像，从 `train_mask_dir` 加载标签（如果有）
2. **SAM2 初始化**：加载 SAM2 模型和权重
3. **环境创建**：创建候选选择环境
4. **PPO 训练**：每个 episode 随机采样一个图像，策略学习选择/融合候选以最大化奖励

### 监控训练

使用 TensorBoard：

```bash
tensorboard --logdir ./logs
```

然后访问 `http://localhost:6006`

### 训练输出

- **日志目录**：`./logs`（TensorBoard 日志）
- **模型保存**：`./checkpoints`（每 N 步保存一次，最终模型）

---

## 推理

### 基本推理命令

```bash
cd /home/ubuntu/sam+RL

python inference.py \
    --model ./checkpoints/ppo_candidate_selector_final.zip \
    --input_dir /path/to/test/images \
    --output_dir ./results \
    --visualize \
    --save_metrics
```

### 参数说明

- `--model`：训练好的模型路径（`.zip` 文件）
- `--input_dir`：输入图像目录
- `--output_dir`：输出目录（掩膜和可视化）
- `--gt_dir`（可选）：真实标签目录，用于评估
- `--visualize`：是否保存可视化结果
- `--save_metrics`：是否保存评估指标到 `metrics.txt`

### 输出

- `{name}.png`：预测掩膜（灰度图，0=背景，255=前景）
- `{name}_vis.png`：可视化（蓝色=预测，绿色边界=GT）
- `metrics.txt`：评估指标（如果提供了 GT）

---

## 示例

### 1. 在 StenUNet 测试集上训练

修改 `config/default.yaml`：

```yaml
data:
  train_image_dir: "/home/ubuntu/StenUNet/dataset_test/raw"
  train_mask_dir: null  # 无标注（使用 vesselness 奖励）
```

然后训练：

```bash
python train.py
```

### 2. 在标注数据上训练

假设你有标注数据：

```yaml
data:
  train_image_dir: "/path/to/images"
  train_mask_dir: "/path/to/masks"
```

然后训练：

```bash
python train.py
```

### 3. 推理并评估

```bash
python inference.py \
    --model ./checkpoints/ppo_candidate_selector_final.zip \
    --input_dir /path/to/test/images \
    --output_dir ./results \
    --gt_dir /path/to/test/masks \
    --visualize \
    --save_metrics
```

---

## 评估指标

- **IoU**（Intersection over Union）：重叠度
- **Dice**：2 * |A ∩ B| / (|A| + |B|)
- **clDice**（centerline Dice）：中心线（骨架）的 Dice，用于评估细长结构
- **拓扑惩罚**：连通组件数（越少越好）
- **Vesselness**：Frangi 血管性覆盖率（无标注时使用）

---

## 课程学习与调优建议

### 1. 从简单数据开始

- 先用对比度高、血管清晰的图像训练
- 逐步加入低对比、噪声、复杂分叉的图像

### 2. 奖励函数调优

- 如果过度碎片化：增加 `topology_weight`
- 如果假阳性过多：增加 `false_positive_penalty`
- 如果中心线不连续：增加 `cldice_weight`

### 3. 动作成本

- `action_cost` 越大（负值），策略越倾向于快速终止
- 平衡探索（多采样）与效率（少动作）

### 4. 候选生成策略

- 调整 `num_initial_points` 和 `mask_thresholds`
- 增加候选多样性有助于策略学习，但也增加计算开销

---

## 下一步（阶段 B/C）

本项目实现了**阶段 A：候选选择器**。后续阶段：

- **阶段 B**：Prompt 决策器（RL 直接控制点/框的位置，而非仅选择候选）
- **阶段 C**：视频与时序一致性（使用 `SAM2VideoPredictor`，跨帧传播与记忆）

---

## 故障排查

### 1. CUDA out of memory

- 减小 `ppo.n_steps` 或 `ppo.batch_size`
- 启用 `sam2.use_half_precision: true`
- 减小 `env.image_size`

### 2. 奖励始终很低

- 检查图像格式（是否需要反转、预处理）
- 检查 GT 标签是否正确加载
- 降低 `action_cost` 的惩罚

### 3. SAM2 加载失败

- 确保 `sam2` 仓库已正确安装（`pip install -e .`）
- 检查 `model_cfg` 路径是否正确（相对于 `/home/ubuntu/sam2/`）
- 检查权重文件是否存在

---

## 参考文档

- [SAM2 官方仓库](https://github.com/facebookresearch/sam2)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [SAM+强化学习思路文档](/home/ubuntu/SAM+强化学习思路.md)

---

## 许可证

本项目代码遵循 MIT 许可证。  
SAM2 遵循 Apache 2.0 许可证。

---

## 联系与贡献

如有问题或建议，请提交 issue 或 pull request。

**作者**：基于 SAM2 与强化学习的血管狭窄分割研究团队  
**日期**：2025-11-03

