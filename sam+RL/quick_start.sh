#!/bin/bash
# 快速开始脚本：配置和测试环境

echo "========================================"
echo "SAM2 + RL 环境配置与测试"
echo "========================================"
echo ""

# 1. 检查 Python 版本
echo "1. 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$python_version >= 3.10" | bc -l) -eq 1 ]]; then
    echo "✓ Python 版本: $(python3 --version)"
else
    echo "✗ Python 版本过低（需要 >= 3.10）"
    exit 1
fi
echo ""

# 2. 检查 PyTorch 和 CUDA
echo "2. 检查 PyTorch 和 CUDA..."
python3 -c "import torch; print(f'✓ PyTorch 版本: {torch.__version__}'); print(f'✓ CUDA 可用: {torch.cuda.is_available()}'); print(f'✓ CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>&1
if [ $? -ne 0 ]; then
    echo "✗ PyTorch 未安装或导入失败"
    echo "  请先安装 PyTorch: https://pytorch.org/get-started/locally/"
    exit 1
fi
echo ""

# 3. 检查 SAM2
echo "3. 检查 SAM2..."
python3 -c "import sys; sys.path.insert(0, '/home/ubuntu/sam2'); from sam2.build_sam import build_sam2; print('✓ SAM2 已安装')" 2>&1
if [ $? -ne 0 ]; then
    echo "✗ SAM2 未安装"
    echo "  请先安装 SAM2:"
    echo "    cd /home/ubuntu/sam2"
    echo "    pip install -e ."
    exit 1
fi
echo ""

# 4. 检查 SAM2 权重
echo "4. 检查 SAM2 权重..."
if [ -f "/home/ubuntu/sam2.1_hiera_large.pt" ]; then
    echo "✓ SAM2 权重文件存在: /home/ubuntu/sam2.1_hiera_large.pt"
else
    echo "✗ SAM2 权重文件不存在: /home/ubuntu/sam2.1_hiera_large.pt"
    echo "  请下载权重文件或修改配置文件中的路径"
    exit 1
fi
echo ""

# 5. 安装项目依赖
echo "5. 安装项目依赖..."
if [ ! -f "/home/ubuntu/sam+RL/requirements.txt" ]; then
    echo "✗ requirements.txt 不存在"
    exit 1
fi

pip install -r /home/ubuntu/sam+RL/requirements.txt -q
if [ $? -eq 0 ]; then
    echo "✓ 依赖安装完成"
else
    echo "✗ 依赖安装失败"
    exit 1
fi
echo ""

# 6. 检查数据目录
echo "6. 检查数据目录..."
data_dir="/home/ubuntu/StenUNet/dataset_test/raw"
if [ -d "$data_dir" ]; then
    num_images=$(find "$data_dir" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
    echo "✓ 数据目录存在: $data_dir"
    echo "  图像数量: $num_images"
else
    echo "⚠️  默认数据目录不存在: $data_dir"
    echo "  请修改配置文件中的 data.train_image_dir"
fi
echo ""

# 7. 测试环境
echo "7. 测试环境（加载 SAM2 和环境）..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/sam+RL')
sys.path.insert(0, '/home/ubuntu/sam2')

try:
    from models.sam2_wrapper import SAM2CandidateGenerator
    from rewards.reward_functions import RewardCalculator
    from env.candidate_selection_env import CandidateSelectionEnv
    import yaml
    
    # 加载配置
    with open('/home/ubuntu/sam+RL/config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("✓ 所有模块导入成功")
    print("✓ 配置文件加载成功")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "✓ 环境测试通过"
else
    echo "✗ 环境测试失败"
    exit 1
fi
echo ""

# 8. 创建输出目录
echo "8. 创建输出目录..."
mkdir -p /home/ubuntu/sam+RL/logs
mkdir -p /home/ubuntu/sam+RL/checkpoints
mkdir -p /home/ubuntu/sam+RL/results
echo "✓ 输出目录已创建"
echo ""

echo "========================================"
echo "✓ 环境配置完成！"
echo "========================================"
echo ""
echo "下一步："
echo "  1. 修改配置文件（如需）: nano /home/ubuntu/sam+RL/config/default.yaml"
echo "  2. 开始训练: cd /home/ubuntu/sam+RL && python train.py"
echo "  3. 监控训练: tensorboard --logdir ./logs"
echo ""

