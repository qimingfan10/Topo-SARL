#!/bin/bash
# 使用 version_125 checkpoint 生成可视化视频

set -e

echo "🚀 使用 version_125 checkpoint 生成可视化..."
echo ""

cd /home/ubuntu/RL4Seg3D

# Checkpoint 路径
CKPT_PATH="/home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_RewardUnets3D_my_organized_dataset_3d/version_125/checkpoints/last.ckpt"

# 验证 checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: 找不到 checkpoint: $CKPT_PATH"
    exit 1
fi

echo "✅ Checkpoint 已找到:"
ls -lh "$CKPT_PATH"
echo ""

# 创建输出目录
OUTPUT_DIR="visualization_outputs/version_125"
echo "📁 创建输出目录: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{predictions,videos}
echo ""

# 输入数据
INPUT_PATH="/home/ubuntu/my_organized_dataset/img"
NUM_FILES=$(find "$INPUT_PATH" -name "*.nii*" 2>/dev/null | wc -l)
echo "📊 将处理 $NUM_FILES 个文件"
echo "⏱️  预计时间: $((NUM_FILES * 2)) - $((NUM_FILES * 4)) 分钟"
echo ""

# 步骤1: 运行推理
echo "🔮 步骤 1/2: 运行模型推理..."
echo "============================================"
python3 rl4seg3d/predict_3d.py \
  input_path="$INPUT_PATH" \
  output_path="./$OUTPUT_DIR/predictions" \
  ckpt_path="$CKPT_PATH"

if [ $? -ne 0 ]; then
    echo "❌ 推理失败"
    exit 1
fi

echo ""
echo "✅ 推理完成"
echo ""

# 步骤2: 转换为视频
echo "🎥 步骤 2/2: 转换为MP4视频..."
echo "============================================"

# 检查预测结果位置（可能在 rewardDS 子目录）
PRED_DIR="$OUTPUT_DIR/predictions"
if [ -d "$PRED_DIR/rewardDS" ]; then
    echo "✓ 找到 rewardDS 目录，使用 reward-dataset 模式..."
    python3 scripts/nifti_to_mp4.py \
      -i "$PRED_DIR/rewardDS" \
      -o "$OUTPUT_DIR/videos" \
      --reward-dataset \
      --fps 5
else
    echo "✓ 使用标准批量转换模式..."
    python3 scripts/nifti_to_mp4.py \
      -i "$PRED_DIR" \
      -o "$OUTPUT_DIR/videos" \
      --batch \
      --fps 5 \
      --width 800
fi

if [ $? -ne 0 ]; then
    echo "❌ 视频转换失败"
    exit 1
fi

echo ""
echo "🎉 全部完成！"
echo "============================================"
echo ""
echo "📊 结果统计:"
NUM_PREDS=$(ls $OUTPUT_DIR/predictions/*.nii.gz 2>/dev/null | wc -l)
NUM_VIDEOS=$(ls $OUTPUT_DIR/videos/*.mp4 2>/dev/null | wc -l)
echo "  - 预测结果: $NUM_PREDS 个文件"
echo "  - 视频文件: $NUM_VIDEOS 个文件"
echo ""
echo "📁 输出位置:"
echo "  - 预测: $OUTPUT_DIR/predictions/"
echo "  - 视频: $OUTPUT_DIR/videos/"
echo ""
echo "🎬 查看视频列表:"
ls -lh "$OUTPUT_DIR/videos"/*.mp4 2>/dev/null | head -10
echo ""
echo "📥 下载命令（在本地运行）:"
echo "   scp -r ubuntu@YOUR_SERVER:/home/ubuntu/RL4Seg3D/$OUTPUT_DIR/videos/ ./"
echo ""
echo "============================================"

