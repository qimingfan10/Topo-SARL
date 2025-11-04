#!/bin/bash
# 对比基线模型和改进模型的效果

set -e

echo "🔬 生成基线模型 vs 改进模型对比..."
echo ""

cd /home/ubuntu/RL4Seg3D

# Checkpoint 路径
BASELINE_CKPT="./data/checkpoints/rl4seg3d_slim.ckpt"
IMPROVED_CKPT="/home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_RewardUnets3D_my_organized_dataset_3d/version_125/checkpoints/last.ckpt"

# 验证 checkpoints
echo "🔍 验证 checkpoint 文件..."
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "❌ 找不到基线模型: $BASELINE_CKPT"
    exit 1
fi
if [ ! -f "$IMPROVED_CKPT" ]; then
    echo "❌ 找不到改进模型: $IMPROVED_CKPT"
    exit 1
fi

echo "✅ 基线模型: $(ls -lh $BASELINE_CKPT | awk '{print $5}')"
echo "✅ 改进模型: $(ls -lh $IMPROVED_CKPT | awk '{print $5}')"
echo ""

# 输入数据
INPUT_PATH="/home/ubuntu/my_organized_dataset/img"
NUM_FILES=$(find "$INPUT_PATH" -name "*.nii*" 2>/dev/null | wc -l)

echo "📊 将处理 $NUM_FILES 个文件 x 2 个模型"
echo "⏱️  预计总时间: $((NUM_FILES * 4)) - $((NUM_FILES * 8)) 分钟"
echo ""

read -p "是否继续? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 创建输出目录
mkdir -p visualization_outputs/comparison/{baseline,improved}/{predictions,videos}

# ========================================
# 基线模型预测
# ========================================
echo ""
echo "📍 步骤 1/4: 基线模型推理..."
echo "============================================"
python3 rl4seg3d/predict_3d.py \
  input_path="$INPUT_PATH" \
  output_path=./visualization_outputs/comparison/baseline/predictions \
  ckpt_path="$BASELINE_CKPT"

if [ $? -ne 0 ]; then
    echo "❌ 基线模型推理失败"
    exit 1
fi
echo "✅ 基线模型推理完成"

# ========================================
# 基线模型转视频
# ========================================
echo ""
echo "📍 步骤 2/4: 基线模型转视频..."
echo "============================================"

BASELINE_PRED="visualization_outputs/comparison/baseline/predictions"
if [ -d "$BASELINE_PRED/rewardDS" ]; then
    python3 scripts/nifti_to_mp4.py \
      -i "$BASELINE_PRED/rewardDS" \
      -o visualization_outputs/comparison/baseline/videos \
      --reward-dataset \
      --fps 5
else
    python3 scripts/nifti_to_mp4.py \
      -i "$BASELINE_PRED" \
      -o visualization_outputs/comparison/baseline/videos \
      --batch \
      --fps 5 \
      --width 800
fi

if [ $? -ne 0 ]; then
    echo "❌ 基线模型转视频失败"
    exit 1
fi
echo "✅ 基线模型视频生成完成"

# ========================================
# 改进模型预测
# ========================================
echo ""
echo "📍 步骤 3/4: 改进模型推理..."
echo "============================================"
python3 rl4seg3d/predict_3d.py \
  input_path="$INPUT_PATH" \
  output_path=./visualization_outputs/comparison/improved/predictions \
  ckpt_path="$IMPROVED_CKPT"

if [ $? -ne 0 ]; then
    echo "❌ 改进模型推理失败"
    exit 1
fi
echo "✅ 改进模型推理完成"

# ========================================
# 改进模型转视频
# ========================================
echo ""
echo "📍 步骤 4/4: 改进模型转视频..."
echo "============================================"

IMPROVED_PRED="visualization_outputs/comparison/improved/predictions"
if [ -d "$IMPROVED_PRED/rewardDS" ]; then
    python3 scripts/nifti_to_mp4.py \
      -i "$IMPROVED_PRED/rewardDS" \
      -o visualization_outputs/comparison/improved/videos \
      --reward-dataset \
      --fps 5
else
    python3 scripts/nifti_to_mp4.py \
      -i "$IMPROVED_PRED" \
      -o visualization_outputs/comparison/improved/videos \
      --batch \
      --fps 5 \
      --width 800
fi

if [ $? -ne 0 ]; then
    echo "❌ 改进模型转视频失败"
    exit 1
fi
echo "✅ 改进模型视频生成完成"

# ========================================
# 统计结果
# ========================================
echo ""
echo "🎉 全部完成！"
echo "============================================"
echo ""
echo "📊 结果统计:"
BASELINE_VIDEOS=$(ls visualization_outputs/comparison/baseline/videos/*.mp4 2>/dev/null | wc -l)
IMPROVED_VIDEOS=$(ls visualization_outputs/comparison/improved/videos/*.mp4 2>/dev/null | wc -l)
echo "  - 基线模型视频: $BASELINE_VIDEOS 个"
echo "  - 改进模型视频: $IMPROVED_VIDEOS 个"
echo ""
echo "📁 输出位置:"
echo "  - 基线模型: visualization_outputs/comparison/baseline/videos/"
echo "  - 改进模型: visualization_outputs/comparison/improved/videos/"
echo ""
echo "🎬 查看文件列表:"
echo ""
echo "基线模型视频:"
ls -lh visualization_outputs/comparison/baseline/videos/*.mp4 2>/dev/null | head -5
echo ""
echo "改进模型视频:"
ls -lh visualization_outputs/comparison/improved/videos/*.mp4 2>/dev/null | head -5
echo ""
echo "📥 下载命令（在本地运行）:"
echo "   scp -r ubuntu@YOUR_SERVER:/home/ubuntu/RL4Seg3D/visualization_outputs/comparison/ ./"
echo ""
echo "============================================"
echo ""
echo "💡 提示: 同时播放对应的视频文件，对比分割效果！"

