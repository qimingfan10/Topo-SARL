#!/bin/bash
# 一键生成训练结果可视化视频

set -e  # 遇到错误立即退出

echo "🎬 开始生成可视化视频..."
echo ""

# 切换到项目目录
cd /home/ubuntu/RL4Seg3D

# 创建输出目录
echo "📁 创建输出目录..."
mkdir -p visualization_outputs/{predictions,videos}
echo "✅ 目录创建完成"
echo ""

# 检查checkpoint
CKPT_PATH="./data/checkpoints/rl4seg3d_slim.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: 找不到checkpoint文件: $CKPT_PATH"
    echo "请检查文件路径或指定其他checkpoint"
    exit 1
fi
echo "✅ Checkpoint已找到: $CKPT_PATH"
echo ""

# 检查输入数据
INPUT_PATH="/home/ubuntu/my_organized_dataset/img"
if [ ! -d "$INPUT_PATH" ]; then
    echo "❌ 错误: 找不到输入数据目录: $INPUT_PATH"
    exit 1
fi

# 统计文件数量
NUM_FILES=$(find "$INPUT_PATH" -name "*.nii*" | wc -l)
echo "✅ 找到 $NUM_FILES 个输入文件"
echo ""

# 询问用户是否继续
echo "⚠️  这将处理 $NUM_FILES 个文件，预计需要 $((NUM_FILES * 2)) - $((NUM_FILES * 4)) 分钟"
echo ""
read -p "是否继续? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 步骤1: 运行推理
echo "🔮 步骤 1/2: 运行模型推理..."
echo "-----------------------------------"
python3 rl4seg3d/predict_3d.py \
  input_path="$INPUT_PATH" \
  output_path=./visualization_outputs/predictions \
  ckpt_path="$CKPT_PATH"

if [ $? -ne 0 ]; then
    echo "❌ 推理失败，请检查错误信息"
    exit 1
fi
echo ""
echo "✅ 推理完成"
echo ""

# 步骤2: 转换为视频
echo "🎥 步骤 2/2: 转换为MP4视频..."
echo "-----------------------------------"
python3 scripts/nifti_to_mp4.py \
  -i visualization_outputs/predictions \
  -o visualization_outputs/videos \
  --batch \
  --fps 5 \
  --width 800

if [ $? -ne 0 ]; then
    echo "❌ 视频转换失败，请检查错误信息"
    exit 1
fi
echo ""
echo "✅ 视频转换完成"
echo ""

# 显示结果
echo "🎉 全部完成！"
echo "============================================"
echo ""
echo "📊 生成的文件:"
echo "-----------------------------------"
echo "预测结果 (NIfTI): visualization_outputs/predictions/"
ls -lh visualization_outputs/predictions/*.nii.gz 2>/dev/null | wc -l | xargs echo "  - 共"
echo ""
echo "视频文件 (MP4): visualization_outputs/videos/"
ls -lh visualization_outputs/videos/*.mp4 2>/dev/null | head -10
echo ""

NUM_VIDEOS=$(ls visualization_outputs/videos/*.mp4 2>/dev/null | wc -l)
echo "✅ 共生成 $NUM_VIDEOS 个视频文件"
echo ""
echo "📥 下载到本地观看（在本地终端运行）:"
echo "   scp -r ubuntu@YOUR_SERVER:/home/ubuntu/RL4Seg3D/visualization_outputs/videos/ ./"
echo ""
echo "============================================"

