# Topo-SARL: Topological Segmentation with Reinforcement Learning

这是一个结合了 SAM2（Segment Anything Model 2）和强化学习（Reinforcement Learning）的医学图像分割项目集合。

## 项目概述

本项目包含三个主要模块：

1. **sam+RL**: 结合 SAM2 与 PPO 强化学习算法，用于血管狭窄分割任务
2. **RL4Seg3D**: 基于强化学习的 3D 超声心动图分割框架，用于领域自适应
3. **sam2**: Meta AI 的 Segment Anything Model 2 基础模型

## 目录结构

```
Topo-SARL/
├── sam+RL/              # SAM2 + 强化学习：血管狭窄分割
│   ├── config/         # 配置文件
│   ├── models/         # 模型定义（SAM2 包装器、PPO 策略）
│   ├── env/            # RL 环境（候选选择器）
│   ├── rewards/        # 奖励函数
│   ├── utils/          # 工具函数
│   ├── train.py        # 训练脚本
│   └── inference.py    # 推理脚本
│
├── RL4Seg3D/           # 强化学习 3D 分割框架
│   ├── rl4seg3d/       # 核心 RL 模块
│   ├── patchless-nnUnet/  # Patchless nnUnet 实现
│   ├── vital/          # Vital 框架
│   ├── training/       # 训练相关代码
│   └── scripts/        # 工具脚本
│
├── sam2/               # SAM2 基础模型
│   ├── sam2/           # SAM2 核心代码
│   ├── training/       # 训练代码
│   ├── demo/           # Web 演示
│   ├── notebooks/      # Jupyter 示例
│   └── tools/          # 工具脚本
│
├── requirements.txt    # Python 环境依赖
└── README.md           # 本文档
```

## 环境要求

- Python >= 3.10
- PyTorch >= 2.5.1
- CUDA（用于 GPU 加速）

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/qimingfan10/Topo-SARL.git
cd Topo-SARL
```

### 2. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt
```

### 3. 安装 SAM2

```bash
cd sam2
pip install -e .
```

如果需要运行示例 notebooks：

```bash
pip install -e ".[notebooks]"
```

### 4. 安装 RL4Seg3D

```bash
cd ../RL4Seg3D
pip install -e .
```

## 项目详细说明

### sam+RL: 血管狭窄分割

**项目简介**: 结合 SAM2 与 PPO 强化学习算法，用于血管狭窄分割任务。

**核心思路**:
- **SAM2**: 生成多个候选掩膜（通过不同的提示点和阈值）
- **PPO RL**: 学习如何从候选中选择/融合，以最大化分割质量
- **奖励函数**: IoU、clDice（中心线 Dice）、拓扑约束、vesselness 等

**快速开始**:
```bash
cd sam+RL

# 训练模型
python train.py --config config/default.yaml

# 推理
python inference.py --input <input_path> --output <output_path>
```

详细文档请参考: [sam+RL/README.md](sam+RL/README.md)

### RL4Seg3D: 3D 强化学习分割框架

**项目简介**: 基于强化学习的 3D 超声心动图分割框架，用于领域自适应。

**核心特点**:
- 单时间步分割的强化学习定义
- 迭代式三步骤循环：预测 → 训练奖励网络 → 微调分割策略
- 支持 TorchScript 推理和 Docker 容器部署

**快速开始**:
```bash
cd RL4Seg3D

# TorchScript 推理
python torchscript_predict_3d.py --input <INPUT_PATH> --output <OUTPUT_PATH>

# 完整训练
python rl4seg3d/auto_iteration.py
```

详细文档请参考: [RL4Seg3D/README.md](RL4Seg3D/README.md)

### sam2: Segment Anything Model 2

**项目简介**: Meta AI 的 Segment Anything Model 2 基础模型，用于图像和视频的可提示视觉分割。

**核心特点**:
- 支持图像和视频分割
- 实时视频处理（流式内存）
- 多个预训练模型检查点

**快速开始**:
```bash
cd sam2

# 使用示例 notebook
jupyter notebook notebooks/image_predictor_example.ipynb
```

详细文档请参考: [sam2/README.md](sam2/README.md)

## 注意事项

1. **大文件处理**: 
   - 所有 checkpoint 文件已通过 `.gitignore` 排除，不会被上传到仓库
   - 日志文件（`.log`）和图片文件（`.png`, `.jpg` 等）也被排除
   - notebooks 中的视频文件已排除

2. **依赖管理**:
   - `requirements.txt` 包含当前环境的完整依赖列表
   - 某些本地开发依赖（如 `-e /home/ubuntu/RL4Seg3D/vital`）可能需要根据实际环境调整

3. **GPU 要求**:
   - 建议使用 GPU 进行训练和推理
   - 确保安装了正确的 CUDA 版本和 PyTorch 版本

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

请参考各个子项目的 LICENSE 文件：
- `sam2/LICENSE`
- `RL4Seg3D/LICENSE`

## 引用

如果您使用了本项目，请引用相关论文：

### SAM2
```
@article{sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2406.xxxxx},
  year={2024}
}
```

### RL4Seg3D
```
@article{judge2024domain,
  title={Domain Adaptation of Echocardiography Segmentation Via Reinforcement Learning},
  author={Judge, Arnaud and others},
  journal={MICCAI},
  year={2024}
}
```

## 联系方式

如有问题，请通过 GitHub Issues 联系。

---

**最后更新**: 2024年11月

