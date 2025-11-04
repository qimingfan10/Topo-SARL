#!/bin/bash
# 使用最新训练的checkpoint生成可视化视频

set -e

echo "🚀 使用最新训练模型生成可视化..."
echo ""

cd /home/ubuntu/RL4Seg3D

# 查找最新的checkpoint
echo "🔍 搜索最新的checkpoint..."
LATEST_CKPT=$(find outputs lightning_logs -name "*.ckpt" -type f 2>/dev/null | sort -t/ -k4 -r | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "❌ 错误: 找不到训练生成的checkpoint"
    echo ""
    echo "可能原因："
    echo "  1. 训练还在进行中，checkpoint尚未保存"
    echo "  2. checkpoint保存在其他位置"
    echo ""
    echo "💡 建议："
    echo "  - 等待训练完成后再运行此脚本"
    echo "  - 或使用预训练模型: ./一键生成可视化.sh"
    echo ""
    echo "🔍 手动查找checkpoint:"
    echo "   find . -name '*.ckpt' | grep -v __pycache__"
    exit 1
fi

echo "✅ 找到最新checkpoint:"
echo "   $LATEST_CKPT"
ls -lh "$LATEST_CKPT"
echo ""

# 创建输出目录（使用时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="visualization_outputs/improved_${TIMESTAMP}"
echo "📁 创建输出目录: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{predictions,videos}
echo ""

# 输入数据路径
INPUT_PATH="/home/ubuntu/my_organized_dataset/img"
NUM_FILES=$(find "$INPUT_PATH" -name "*.nii*" 2>/dev/null | wc -l)
echo "📊 将处理 $NUM_FILES 个文件"
echo "⏱️  预计时间: $((NUM_FILES * 2)) - $((NUM_FILES * 4)) 分钟"
echo ""

# 询问确认
read -p "是否继续? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 运行推理
echo ""
echo "🔮 运行模型推理..."
echo "============================================"
python3 rl4seg3d/predict_3d.py \
  input_path="$INPUT_PATH" \
  output_path="./$OUTPUT_DIR/predictions" \
  ckpt_path="$LATEST_CKPT"

if [ $? -ne 0 ]; then
    echo "❌ 推理失败"
    exit 1
fi

echo ""
echo "✅ 推理完成"
echo ""

# 转换为视频
echo "🎥 转换为MP4视频..."
echo "============================================"
python3 scripts/nifti_to_mp4.py \
  -i "$OUTPUT_DIR/predictions" \
  -o "$OUTPUT_DIR/videos" \
  --batch \
  --fps 5 \
  --width 800

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
echo ""
echo "💡 提示: 如需对比基线模型，运行 ./一键生成可视化.sh"

