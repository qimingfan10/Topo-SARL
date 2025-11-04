#!/bin/bash
# 使用峰值性能最好的模型 (version_264, 峰值Reward=0.5970) 生成可视化视频

set -e

echo "========================================================================"
echo "🏆 使用最佳模型生成可视化视频"
echo "========================================================================"
echo ""
echo "📊 模型信息:"
echo "   版本: version_264"
echo "   峰值Reward: 0.5970 (step 129)"
echo "   Entropy: 0.03"
echo "   学习率: 0.001"
echo "   梯度健康: ✅"
echo "   策略更新: ✅"
echo ""
echo "========================================================================"

# 配置
CKPT_PATH="/home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_RewardUnets3D_my_organized_dataset_3d/version_264/checkpoints/last.ckpt"
OUTPUT_BASE="/home/ubuntu/visualization_outputs/version_264_best"
DATA_DIR="/home/ubuntu/my_organized_dataset_3d/rewardDS/imagesVal"

# 检查checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint不存在: $CKPT_PATH"
    exit 1
fi

echo ""
echo "✅ Checkpoint已找到"
echo ""

# 步骤1: 生成预测
echo "========================================================================"
echo "📝 步骤1/3: 生成模型预测"
echo "========================================================================"

cd /home/ubuntu/RL4Seg3D

python3 rl4seg3d/predict_3d.py \
    ckpt_path="$CKPT_PATH" \
    model=ppo_3d \
    actor=3d_ac_unet \
    dataset=my_organized_3d \
    output_path="${OUTPUT_BASE}/predictions" \
    batch_size=1

echo ""
echo "✅ 预测完成"
echo ""

# 步骤2: 查找预测文件
echo "========================================================================"
echo "📝 步骤2/3: 检查预测文件"
echo "========================================================================"

# 检查两个可能的位置
PRED_DIR="${OUTPUT_BASE}/predictions"
if [ -d "${PRED_DIR}/rewardDS" ]; then
    PRED_DIR="${PRED_DIR}/rewardDS"
    echo "✅ 找到预测文件目录: $PRED_DIR"
else
    echo "⚠️  未找到rewardDS子目录，使用主目录"
fi

NUM_PREDICTIONS=$(find "$PRED_DIR" -name "*.nii.gz" -type f 2>/dev/null | wc -l)

if [ "$NUM_PREDICTIONS" -eq 0 ]; then
    echo ""
    echo "❌ 未找到预测文件！"
    echo "   搜索目录: $PRED_DIR"
    echo ""
    echo "📂 当前目录结构:"
    ls -lR "${OUTPUT_BASE}" 2>/dev/null | head -30
    exit 1
fi

echo "✅ 找到 $NUM_PREDICTIONS 个预测文件"
echo ""

# 步骤3: 转换为视频
echo "========================================================================"
echo "📝 步骤3/3: 转换为MP4视频"
echo "========================================================================"

VIDEO_OUTPUT_DIR="${OUTPUT_BASE}/videos"
mkdir -p "$VIDEO_OUTPUT_DIR"

python3 scripts/nifti_to_mp4.py \
    --prediction-dir "$PRED_DIR" \
    --ground-truth-dir "$DATA_DIR" \
    --output-dir "$VIDEO_OUTPUT_DIR" \
    --reward-dataset

echo ""
echo "========================================================================"
echo "🎉 完成！"
echo "========================================================================"
echo ""
echo "📂 输出位置:"
echo "   预测文件: $PRED_DIR"
echo "   视频文件: $VIDEO_OUTPUT_DIR"
echo ""
echo "📊 生成的视频数量:"
ls -lh "$VIDEO_OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l
echo ""
echo "🎬 查看视频:"
echo "   ls -lh $VIDEO_OUTPUT_DIR/*.mp4"
echo ""
echo "========================================================================"

