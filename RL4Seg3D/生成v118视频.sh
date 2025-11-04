#!/bin/bash
# 生成version_118（pos_weight修复后）的预测视频

set -e

echo "========================================================================"
echo "🎬 生成 version_118 (修复后模型) 的预测视频"
echo "========================================================================"
echo ""

cd /home/ubuntu/RL4Seg3D

# 配置
CKPT_PATH="/home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_icardio3d/version_118/checkpoints/last.ckpt"
OUTPUT_DIR="visualization_outputs/final_v118"
DATA_DIR="/home/ubuntu/my_organized_dataset_3d/rewardDS/imagesVal"

echo "📋 配置信息:"
echo "   Checkpoint: $CKPT_PATH"
echo "   数据目录: $DATA_DIR"
echo "   输出目录: $OUTPUT_DIR"
echo ""

# 检查checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint不存在: $CKPT_PATH"
    echo ""
    echo "可用的checkpoint:"
    find /home/ubuntu/my_rl4seg3d_logs -name "version_118" -type d 2>/dev/null
    exit 1
fi

echo "✅ Checkpoint已找到"
ls -lh "$CKPT_PATH"
echo ""

# 检查数据
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 数据目录不存在: $DATA_DIR"
    echo ""
    echo "尝试查找其他数据目录..."
    DATA_DIR="/home/ubuntu/RL4Seg3D/visualization_outputs/version_125/predictions/rewardDS/images"
    if [ -d "$DATA_DIR" ]; then
        echo "✅ 找到数据目录: $DATA_DIR"
    else
        echo "❌ 未找到可用的数据目录"
        exit 1
    fi
fi

NUM_FILES=$(ls "$DATA_DIR"/*.nii.gz 2>/dev/null | wc -l)
echo "📊 找到 $NUM_FILES 个测试图像"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR/predictions"
mkdir -p "$OUTPUT_DIR/videos"

echo "========================================================================"
echo "📝 步骤1/3: 生成预测"
echo "========================================================================"
echo ""

# 运行预测
python3 rl4seg3d/predict_3d.py \
    input_path="$DATA_DIR" \
    output_path="$OUTPUT_DIR/predictions" \
    ckpt_path="$CKPT_PATH" \
    2>&1 | tee "$OUTPUT_DIR/predict.log"

PREDICT_EXIT=$?

if [ $PREDICT_EXIT -ne 0 ]; then
    echo ""
    echo "❌ 预测失败，退出码: $PREDICT_EXIT"
    echo ""
    echo "查看日志: $OUTPUT_DIR/predict.log"
    exit 1
fi

echo ""
echo "✅ 预测完成"
echo ""

# 查找预测文件
echo "========================================================================"
echo "📝 步骤2/3: 检查预测文件"
echo "========================================================================"
echo ""

PRED_DIR="$OUTPUT_DIR/predictions"
if [ -d "$PRED_DIR/rewardDS/pred" ]; then
    PRED_DIR="$PRED_DIR/rewardDS/pred"
    echo "✅ 找到预测目录: $PRED_DIR"
elif [ -d "$PRED_DIR/pred" ]; then
    PRED_DIR="$PRED_DIR/pred"
    echo "✅ 找到预测目录: $PRED_DIR"
fi

NUM_PREDS=$(ls "$PRED_DIR"/*.nii.gz 2>/dev/null | wc -l)
echo "   预测文件数: $NUM_PREDS"

if [ $NUM_PREDS -eq 0 ]; then
    echo "❌ 未找到预测文件"
    echo ""
    echo "目录结构:"
    ls -lR "$OUTPUT_DIR/predictions" | head -30
    exit 1
fi

echo ""

# 分析预测质量
echo "========================================================================"
echo "📊 快速质量分析"
echo "========================================================================"
echo ""

python3 << 'PYEOF'
import numpy as np
from pathlib import Path
import nibabel as nib
import sys

pred_dir = Path("$PRED_DIR")
pred_files = list(pred_dir.glob("*.nii.gz"))

if not pred_files:
    print("❌ 未找到预测文件")
    sys.exit(1)

print(f"分析 {len(pred_files)} 个预测文件:\n")

ratios = []
for pred_file in pred_files:
    try:
        pred_data = nib.load(str(pred_file)).get_fdata()
        nonzero_ratio = (pred_data > 0).sum() / pred_data.size
        ratios.append(nonzero_ratio)
        
        if nonzero_ratio > 0.95:
            status = "🚨 全屏mask"
        elif nonzero_ratio > 0.7:
            status = "⚠️  过度分割"
        elif nonzero_ratio < 0.001:
            status = "⚠️  几乎无分割"
        else:
            status = "✅ 正常"
        
        print(f"{pred_file.name}: {nonzero_ratio:.2%} {status}")
    except Exception as e:
        print(f"❌ {pred_file.name}: 读取失败")

if ratios:
    avg_ratio = np.mean(ratios)
    print(f"\n平均非零比例: {avg_ratio:.2%}")
    
    print("\n" + "="*80)
    if avg_ratio < 0.5:
        print("✅✅✅ 修复成功！过度分割显著减少！")
    elif avg_ratio < 0.7:
        print("⚠️  有改善，但仍需优化")
    else:
        print("❌ 仍然过度分割")
    print("="*80)
PYEOF

echo ""

# 生成视频
echo "========================================================================"
echo "📝 步骤3/3: 生成MP4视频"
echo "========================================================================"
echo ""

VIDEO_DIR="$OUTPUT_DIR/videos"

python3 scripts/nifti_to_mp4.py \
    --prediction-dir "$PRED_DIR" \
    --ground-truth-dir "$DATA_DIR" \
    --output-dir "$VIDEO_DIR" \
    --reward-dataset

VIDEO_EXIT=$?

if [ $VIDEO_EXIT -ne 0 ]; then
    echo ""
    echo "❌ 视频生成失败"
    exit 1
fi

echo ""
echo "========================================================================"
echo "🎉 完成！"
echo "========================================================================"
echo ""

NUM_VIDEOS=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)

echo "📊 生成结果:"
echo "   预测文件: $NUM_PREDS 个"
echo "   视频文件: $NUM_VIDEOS 个"
echo ""
echo "📁 输出位置:"
echo "   预测: $PRED_DIR"
echo "   视频: $VIDEO_DIR"
echo ""
echo "🎬 视频列表:"
ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null | head -10
echo ""
echo "对比之前的版本:"
echo "   version_125: /home/ubuntu/RL4Seg3D/visualization_outputs/version_125/videos/"
echo "   version_264: /home/ubuntu/RL4Seg3D/visualization_outputs/improved_v264/videos/"
echo "   version_118: $VIDEO_DIR"
echo ""
echo "========================================================================"

