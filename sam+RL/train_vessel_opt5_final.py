#!/usr/bin/env python3
"""
血管优化循环5：平衡Precision+IoU + 里程碑奖励
这是在SAM2框架下的最后一次奖励优化尝试

核心策略：
1. 平衡precision和IoU权重（50:10）
2. 新增IoU里程碑奖励（鼓励突破）
3. 增强终止时的IoU奖励

成功标准：
- 平均IoU ≥ 5.5%
- 掩膜大小 ≤ 60%
- <20%掩膜比例 ≥ 10%

如果失败，建议改变方向（更换模型或方法）
"""
import sys
import yaml
import numpy as np

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    config_path = '/home/ubuntu/sam+RL/config/stage_b_vessel_opt5.yaml'
    
    class Args:
        config = config_path
    
    print(f"\n{'='*80}")
    print(f"🔧 血管优化循环5（18000步）- 最后一次奖励优化")
    print(f"{'='*80}")
    print(f"\n📊 历史回顾:")
    print(f"  Opt1: IoU 4.43%, 掩膜68.1% （小目标奖励）")
    print(f"  Opt2: IoU 4.26%, 掩膜58.2% （暴力惩罚❌）")
    print(f"  Opt3: IoU 4.75%, 掩膜68.5% （Precision×30）")
    print(f"  Opt4: IoU 4.13%, 掩膜66.6% （Precision×100过强❌）")
    print(f"\n❌ 核心问题:")
    print(f"  1. Precision过强 → IoU下降（Opt4）")
    print(f"  2. 掩膜大小始终60-70%（SAM2限制）")
    print(f"  3. 奖励与目标脱节")
    print(f"\n🎯 Opt5策略（平衡尝试）:")
    print(f"  ")
    print(f"  核心：在Precision和IoU之间找平衡点")
    print(f"  ")
    print(f"  1. Precision权重: 100 → 50 (降低)")
    print(f"  2. Delta IoU权重: 5 → 10 (恢复)")
    print(f"  3. Final IoU权重: 10 → 15 (增强)")
    print(f"  4. 新增IoU里程碑奖励:")
    print(f"     - IoU≥5%: +5分")
    print(f"     - IoU≥6%: +8分")
    print(f"     - IoU≥8%: +15分")
    print(f"     - IoU≥10%: +25分")
    print(f"  ")
    print(f"  原理：同时鼓励precision和IoU提升")
    print(f"\n🎯 成功标准:")
    print(f"  ✅ 平均IoU ≥ 5.5% （比Opt3提升16%）")
    print(f"  ✅ 掩膜大小 ≤ 60% （降低9%）")
    print(f"  ✅ <20%掩膜比例 ≥ 10% （提升47%）")
    print(f"\n⚠️  如果Opt5失败:")
    print(f"  → 停止调参，改变方向！")
    print(f"  → 考虑：UNet、血管专用模型、或SAM2+传统算法")
    print(f"{'='*80}\n")
    
    # 运行训练
    print("开始训练...")
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"🏁 优化循环5完成")
    print(f"{'='*80}\n")
    
    opt5_iou = summary.get("avg_final_iou", 0) * 100
    opt5_best = summary.get("best_iou", 0) * 100
    opt5_len = summary.get("avg_episode_length", 0)
    opt5_reward = summary.get("avg_episode_reward", 0)
    
    print(f"完整结果对比:")
    print(f"┌{'─'*25}┬{'─'*9}┬{'─'*9}┬{'─'*9}┬{'─'*9}┬{'─'*9}┬{'─'*10}┐")
    print(f"│ {'指标':<23} │ {'Opt1':<7} │ {'Opt2':<7} │ {'Opt3':<7} │ {'Opt4':<7} │ {'Opt5':<7} │ {'vs 最佳':<8} │")
    print(f"├{'─'*25}┼{'─'*9}┼{'─'*9}┼{'─'*9}┼{'─'*9}┼{'─'*9}┼{'─'*10}┤")
    best_iou = max(4.43, 4.26, 4.75, 4.13, opt5_iou)
    print(f"│ {'平均IoU':<23} │ {'4.43%':<7} │ {'4.26%':<7} │ {'4.75%':<7} │ {'4.13%':<7} │ {f'{opt5_iou:.2f}%':<7} │ {f'{opt5_iou-best_iou:+.2f}%':<8} │")
    print(f"│ {'掩膜大小':<23} │ {'68.1%':<7} │ {'58.2%':<7} │ {'68.5%':<7} │ {'66.6%':<7} │ {'?':<7} │ {'?':<8} │")
    print(f"│ {'平均奖励':<23} │ {'+0.33':<7} │ {'-22.2':<7} │ {'+243':<7} │ {'+951':<7} │ {f'{opt5_reward:+.0f}':<7} │ {'-':<8} │")
    print(f"└{'─'*25}┴{'─'*9}┴{'─'*9}┴{'─'*9}┴{'─'*9}┴{'─'*9}┴{'─'*10}┘")
    
    # 详细分析
    import json
    metrics_file = 'logs/stage_b_vessel_opt5/final_metrics.json'
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        if 'episode_final_areas' in data:
            areas = np.array(data['episode_f