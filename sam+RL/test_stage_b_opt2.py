#!/usr/bin/env python3
"""
优化循环2：强制最小步数
方案：环境层面禁止前5步terminate
"""
import sys
import yaml

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    # 加载优化配置
    config_path = '/home/ubuntu/sam+RL/config/stage_b_opt2.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建参数对象
    class Args:
        config = config_path
    
    print(f"\n{'='*80}")
    print(f"优化循环2：强制最小步数（10000步）")
    print(f"{'='*80}")
    print(f"\n核心改进:")
    print(f"  1. 环境层面强制：前5步禁止terminate")
    print(f"  2. 过早terminate自动转换为positive动作")
    print(f"  3. 增加min_steps_bonus：0.2 → 0.5")
    print(f"  4. 增加exploration_bonus：0.05 → 0.1")
    print(f"  5. 提高探索系数：ent_coef从0.2 → 0.25")
    print(f"\n目标:")
    print(f"  - 过早终止比例：78.9% → <30%")
    print(f"  - Episode长度：3.08步 → >5步")
    print(f"  - 平均IoU：4.64% → >6%")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"优化循环2完成")
    print(f"{'='*80}")
    print(f"\n结果对比:")
    opt2_iou = f'{summary.get("avg_final_iou", 0)*100:.2f}%'
    opt2_best = f'{summary.get("best_iou", 0)*100:.2f}%'
    opt2_len = f'{summary.get("avg_episode_length", 0):.2f}步'
    opt2_reward = f'{summary.get("avg_episode_reward", 0):+.2f}'
    
    print(f"┌{'─'*40}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┐")
    print(f"│ {'指标':<38} │ {'初步测试':<10} │ {'优化1':<10} │ {'优化2':<10} │ {'改进':<10} │")
    print(f"├{'─'*40}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┤")
    print(f"│ {'平均IoU':<38} │ {'4.59%':<10} │ {'4.64%':<10} │ {opt2_iou:<10} │ {'':<10} │")
    print(f"│ {'最佳IoU':<38} │ {'73.40%':<10} │ {'76.06%':<10} │ {opt2_best:<10} │ {'':<10} │")
    print(f"│ {'Episode长度':<38} │ {'2.79步':<10} │ {'2.98步':<10} │ {opt2_len:<10} │ {'':<10} │")
    print(f"│ {'平均奖励':<38} │ {'+1.04':<10} │ {'+0.26':<10} │ {opt2_reward:<10} │ {'':<10} │")
    print(f"└{'─'*40}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┘")
    
    # 详细分析
    print(f"\n详细分析:")
    
    avg_iou = summary.get('avg_final_iou', 0) * 100
    avg_len = summary.get('avg_episode_length', 0)
    
    # Episode长度检查
    print(f"\n📏 Episode长度:")
    if avg_len >= 5.0:
        print(f"  ✅ 成功！平均{avg_len:.2f}步 >= 5步")
    elif avg_len >= 4.0:
        print(f"  ⚠️  接近目标：平均{avg_len:.2f}步")
    else:
        print(f"  ❌ 未达标：平均{avg_len:.2f}步 < 5步")
    
    # IoU检查
    print(f"\n📊 IoU表现:")
    if avg_iou >= 6.0:
        print(f"  ✅ 达到6%目标！当前{avg_iou:.2f}%")
    elif avg_iou > 4.64:
        print(f"  ⚠️  有提升：{avg_iou:.2f}% > 4.64%")
    else:
        print(f"  ❌ 未提升：{avg_iou:.2f}% ≤ 4.64%")
    
    # 综合评估
    print(f"\n🎯 综合评估:")
    if avg_len >= 5.0 and avg_iou >= 6.0:
        print(f"  🎉 优化成功！Episode长度和IoU都达标！")
        print(f"  建议：继续长期训练（20000-50000步）争取达到10% IoU")
    elif avg_len >= 5.0:
        print(f"  ✅ Episode长度达标！但IoU还需提升")
        print(f"  建议：继续训练并可能调整奖励权重")
    elif avg_iou >= 6.0:
        print(f"  ✅ IoU达标！但Episode长度还不够")
        print(f"  建议：进一步增加min_steps或调整奖励")
    else:
        print(f"  ⚠️  还需继续优化")
        print(f"  建议：分析新的日志，找出问题所在")
    
    print(f"\n{'='*80}\n")

