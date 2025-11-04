#!/bin/bash
# 下一步操作脚本

echo "========================================"
echo "SAM2 + RL 下一步操作指南"
echo "========================================"
echo ""

echo "📋 当前状态:"
echo "  ✅ 系统实现完成"
echo "  ✅ 首次测试完成"
echo "  ⚠️  需要优化奖励函数"
echo ""

echo "🔧 第1步：修改配置文件（必需）"
echo "  编辑 config/default.yaml:"
echo "    - use_gt: false"
echo "    - vesselness_weight: 2.0"
echo "    - action_cost: -0.005"
echo "    - max_steps: 10"
echo ""
echo "  执行: nano config/default.yaml"
echo ""

echo "🧪 第2步：快速测试（5分钟）"
echo "  执行: python3 quick_test.py --timesteps 512"
echo "  检查: logs/quick_test_*/quick_test/summary_report.txt"
echo "  目标: vesselness > 0.1"
echo ""

echo "📊 第3步：查看结果"
echo "  查看报告: cat logs/quick_test_*/quick_test/summary_report.txt"
echo "  查看图表: ls logs/quick_test_*/quick_test/metrics_plot.png"
echo "  TensorBoard: tensorboard --logdir logs/"
echo ""

echo "🚀 第4步：长期训练（后台）"
echo "  执行: nohup python3 train.py > train.log 2>&1 &"
echo "  监控: tail -f train.log"
echo "  停止: pkill -f train.py"
echo ""

echo "📖 详细计划请查看:"
echo "  - TEST_REPORT.md: 测试分析和优化建议"
echo "  - IMPLEMENTATION_SUMMARY.md: 完整实现总结"
echo ""
