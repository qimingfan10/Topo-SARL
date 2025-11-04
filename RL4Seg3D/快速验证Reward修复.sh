#!/bin/bash
# 快速验证Reward修复效果
# pos_weight: 20.0 → 2.0

set -e

echo "========================================================================"
echo "🔧 快速验证Reward修复（pos_weight: 20.0 → 2.0）"
echo "========================================================================"
echo ""
echo "目标: 验证降低pos_weight能否减少过度分割"
echo ""
echo "========================================================================"

cd /home/ubuntu/RL4Seg3D

# 配置
NUM_ITER=1
NUM_EPOCHS=10
OUTPUT_LOG="quick_reward_fix_verification.log"

echo ""
echo "📋 训练配置:"
echo "   迭代次数: $NUM_ITER"
echo "   Epochs: $NUM_EPOCHS"
echo "   pos_weight: 2.0 (修复后)"
echo "   entropy: 0.03"
echo "   学习率: 0.001"
echo ""
echo "========================================================================"

# 检查配置
echo ""
echo "✅ 验证配置修改:"
grep "pos_weight" rl4seg3d/config/reward/rewardunets_3d.yaml

echo ""
echo "========================================================================"
echo "🚀 开始训练..."
echo "========================================================================"

# 运行训练
nohup python3 rl4seg3d/auto_iteration.py \
  num_iter=$NUM_ITER \
  rl_num_epochs=$NUM_EPOCHS \
  > "$OUTPUT_LOG" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > reward_fix_training.pid

echo ""
echo "✅ 训练已启动"
echo "   PID: $TRAIN_PID"
echo "   日志: $OUTPUT_LOG"
echo ""
echo "========================================================================"
echo "📊 监控命令:"
echo "========================================================================"
echo ""
echo "# 查看实时日志"
echo "tail -f $OUTPUT_LOG"
echo ""
echo "# 检查进程"
echo "ps aux | grep auto_iteration"
echo ""
echo "# 查看最新版本"
echo "ls -lth /home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_RewardUnets3D_my_organized_dataset_3d/ | head -5"
echo ""
echo "========================================================================"
echo "⏱️  预计时间: 15-30分钟"
echo ""
echo "训练完成后运行:"
echo "  ./验证修复效果并生成视频.sh"
echo ""
echo "========================================================================"

