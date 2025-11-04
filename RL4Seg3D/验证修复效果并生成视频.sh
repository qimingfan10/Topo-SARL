#!/bin/bash
# 验证Reward修复效果

set -e

echo "========================================================================"
echo "🔍 验证Reward修复效果（pos_weight: 20.0 → 2.0）"
echo "========================================================================"

cd /home/ubuntu/RL4Seg3D

# 找到最新的版本
LATEST_VERSION=$(ls -t /home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_RewardUnets3D_my_organized_dataset_3d/ | grep "version_" | head -1)

if [ -z "$LATEST_VERSION" ]; then
    echo "❌ 未找到最新训练版本"
    exit 1
fi

echo ""
echo "✅ 找到最新版本: $LATEST_VERSION"
echo ""

CKPT_PATH="/home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_RewardUnets3D_my_organized_dataset_3d/$LATEST_VERSION/checkpoints/last.ckpt"
OUTPUT_DIR="/home/ubuntu/RL4Seg3D/visualization_outputs/reward_fixed_$LATEST_VERSION"

# 检查checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint不存在: $CKPT_PATH"
    exit 1
fi

echo "========================================================================"
echo "📝 步骤1: 生成预测"
echo "========================================================================"

python3 rl4seg3d/predict_3d.py \
    ckpt_path="$CKPT_PATH" \
    model=ppo_3d \
    actor=3d_ac_unet \
    dataset=my_organized_3d \
    output_path="${OUTPUT_DIR}/predictions" \
    batch_size=1

echo ""
echo "✅ 预测完成"
echo ""

# 查找预测文件目录
PRED_DIR="${OUTPUT_DIR}/predictions"
if [ -d "${PRED_DIR}/rewardDS" ]; then
    PRED_DIR="${PRED_DIR}/rewardDS"
fi

echo "========================================================================"
echo "📝 步骤2: 分析预测质量"
echo "========================================================================"

python3 << EOF
import numpy as np
from pathlib import Path
import nibabel as nib

pred_dir = Path("${PRED_DIR}/pred")
if not pred_dir.exists():
    print(f"❌ 预测目录不存在: {pred_dir}")
    exit(1)

pred_files = list(pred_dir.glob("*.nii.gz"))
print(f"\n找到 {len(pred_files)} 个预测文件\n")

results = []
for pred_file in pred_files:
    pred_data = nib.load(str(pred_file)).get_fdata()
    nonzero_ratio = (pred_data > 0).sum() / pred_data.size
    results.append(nonzero_ratio)
    print(f"{pred_file.name}: {nonzero_ratio:.4f} ({nonzero_ratio*100:.2f}%)")

avg_ratio = np.mean(results)
print(f"\n平均非零比例: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")

print("\n" + "="*80)
print("🎯 效果评估")
print("="*80)

if avg_ratio < 0.5:
    print(f"""
✅✅✅ 修复成功！

对比:
  修复前 (v125): 91-92% 过度分割
  修复前 (v264): 98% 严重过度分割
  修复后 ({avg_ratio*100:.1f}%): {'正常范围' if avg_ratio < 0.4 else '仍有改进空间'}

pos_weight从20.0降低到2.0显著减少了过度分割！
    """)
elif avg_ratio < 0.7:
    print(f"""
⚠️  部分改善

对比:
  修复前: 91-98% 过度分割
  修复后: {avg_ratio*100:.1f}%

有改善但还不够理想，可能需要:
  1. 进一步降低pos_weight (尝试1.0)
  2. 延长训练时间
  3. 调整其他超参数
    """)
else:
    print(f"""
❌ 仍然过度分割

修复后仍有 {avg_ratio*100:.1f}% 的过度分割

可能需要:
  1. 将pos_weight降到1.0 (无加权)
  2. 检查其他reward计算逻辑
  3. 重新设计reward函数
    """)

EOF

echo ""
echo "========================================================================"
echo "📝 步骤3: 生成可视化视频"
echo "========================================================================"

VIDEO_DIR="${OUTPUT_DIR}/videos"
mkdir -p "$VIDEO_DIR"

# 数据目录
DATA_DIR="/home/ubuntu/my_organized_dataset_3d/rewardDS/imagesVal"
if [ ! -d "$DATA_DIR" ]; then
    # 尝试其他可能的位置
    DATA_DIR="/home/ubuntu/RL4Seg3D/visualization_outputs/version_125/predictions/rewardDS/images"
fi

python3 scripts/nifti_to_mp4.py \
    --prediction-dir "${PRED_DIR}/pred" \
    --ground-truth-dir "$DATA_DIR" \
    --output-dir "$VIDEO_DIR" \
    --reward-dataset

echo ""
echo "========================================================================"
echo "🎉 完成！"
echo "========================================================================"
echo ""
echo "📂 输出位置:"
echo "   预测: ${PRED_DIR}/pred"
echo "   视频: $VIDEO_DIR"
echo ""
echo "🎬 查看视频:"
echo "   ls -lh $VIDEO_DIR/*.mp4"
echo ""
echo "📊 对比之前的版本:"
echo "   version_125: /home/ubuntu/RL4Seg3D/visualization_outputs/version_125/videos/"
echo "   version_264: /home/ubuntu/RL4Seg3D/visualization_outputs/improved_v264/videos/"
echo "   修复后: $VIDEO_DIR"
echo ""
echo "========================================================================"

