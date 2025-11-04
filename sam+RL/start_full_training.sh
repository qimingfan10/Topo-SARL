#!/bin/bash
# 启动完整训练

echo "============================================================"
echo "开始完整训练（100K步）"
echo "============================================================"
echo ""

cd /home/ubuntu/sam+RL

# 显示配置
echo "配置信息："
echo "  总步数: 100,000"
echo "  训练数据: /home/ubuntu/Segment_DATA/orgin_pic (220张)"
echo "  标注数据: /home/ubuntu/Segment_DATA/lab_pic (220张)"
echo "  模式: 有监督训练 (use_gt=true)"
echo "  日志目录: ./logs"
echo "  模型保存: ./checkpoints"
echo ""

# 启动训练
echo "启动训练..."
nohup python3 train.py > logs/full_training.log 2>&1 &

TRAIN_PID=$!
echo ""
echo "✓ 训练已启动（后台运行）"
echo "  进程ID: $TRAIN_PID"
echo ""

echo "监控命令："
echo "  查看日志: tail -f logs/full_training.log"
echo "  查看进度: watch -n 5 'tail -20 logs/full_training.log'"
echo "  TensorBoard: tensorboard --logdir ./logs --port 6006"
echo "  停止训练: kill $TRAIN_PID"
echo ""

echo "预计训练时间: ~1-2小时"
echo ""

