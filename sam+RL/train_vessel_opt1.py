#!/usr/bin/env python3
"""
血管优化循环1：小目标专门优化
- 增加网格精度（48×48）
- 添加掩膜大小惩罚/奖励
- 增强调试信息
"""
import sys
import yaml
import numpy as np

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    config_path = '/home/ubuntu/sam+RL/config/stage_b_vessel_opt1.yaml'
    
    class Args:
        config = config_path
    
    print(f"\n{'='*80}")
    print(f"🔧 血管优化循环1（10000步）")
    print(f"{'='*80}")
    print(f"\n📊 基准性能（真实数据集初测）:")
    print(f"  平均IoU: 3.86%")
    print(f"  最佳IoU: 57.15%")
    print(f"  掩膜过大: 67.36% (目标<10%)")
    print(f"\n🎯 本轮改进:")
    print(f"  1. 增加网格精度: 32×32 → 48×48")
    print(f"  2. 添加掩膜大小惩罚: >20%时惩罚")
    print(f"  3. 添加小掩膜奖励: <10%时奖励")
    print(f"  4. 增强奖励权重: delta_iou×20, final_iou×10")
    print(f"  5. 增加最小步数: 5步 → 7步")
    print(f"  6. 启用详细调试信息")
    print(f"\n预期目标:")
    print(f"  • 平均IoU: 3.86% → 5-6%")
    print(f"  • 掩膜大小: 67.36% → 10-20%")
    print(f"  • Episode长度: >7步")
    print(f"{'='*80}\n")
    
    # 运行训练
    print("开始训练...")
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"🏁 优化循环1完成")
    print(f"{'='*80}")
    
    opt1_iou = summary.get("avg_final_iou", 0) * 100
    opt1_best = summary.get("best_iou", 0) * 100
    opt1_len = summary.get("avg_episode_length", 0)
    opt1_reward = summary.get("avg_episode_reward", 0)
    
    print(f"\n结果对比:")
    print(f"┌{'─'*35}┬{'─'*12}┬{'─'*12}┬{'─'*10}┐")
    print(f"│ {'指标':<33} │ {'基准':<10} │ {'优化1':<10} │ {'变化':<8} │")
    print(f"├{'─'*35}┼{'─'*12}┼{'─'*12}┼{'─'*10}┤")
    print(f"│ {'平均IoU':<33} │ {'3.86%':<10} │ {f'{opt1_iou:.2f}%':<10} │ {f'{opt1_iou-3.86:+.2f}%':<8} │")
    print(f"│ {'最佳IoU':<33} │ {'57.15%':<10} │ {f'{opt1_best:.2f}%':<10} │ {f'{opt1_best-57.15:+.2f}%':<8} │")
    print(f"│ {'Episode长度':<33} │ {'7.42步':<10} │ {f'{opt1_len:.2f}步':<10} │ {f'{opt1_len-7.42:+.2f}':<8} │")
    print(f"│ {'平均奖励':<33} │ {'+2.17':<10} │ {f'{opt1_reward:+.2f}':<10} │ {f'{opt1_reward-2.17:+.2f}':<8} │")
    print(f"└{'─'*35}┴{'─'*12}┴{'─'*12}┴{'─'*10}┘")
    
    # 评估改进效果
    print(f"\n📊 改进评估:")
    improvements = []
    issues = []
    
    if opt1_iou > 3.86:
        delta = opt1_iou - 3.86
        improvements.append(f"  ✅ IoU提升 {delta:.2f}% ({delta/3.86*100:.0f}%)")
    else:
        issues.append(f"  ⚠️  IoU未提升: {opt1_iou:.2f}% ≤ 3.86%")
    
    if opt1_iou >= 5.0:
        improvements.append(f"  🎉 达到5%目标！")
    
    if opt1_len >= 7.0:
        improvements.append(f"  ✅ Episode长度达标 ({opt1_len:.1f}步 ≥ 7步)")
    
    if improvements:
        print("\n改进点:")
        for imp in improvements:
            print(imp)
    
    if issues:
        print("\n需关注:")
        for issue in issues:
            print(issue)
    
    # 加载详细指标进行深度分析
    import json
    metrics_file = 'logs/stage_b_vessel_opt1/final_metrics.json'
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # 分析掩膜大小
        if 'episode_final_areas' in data:
            areas = np.array(data['episode_final_areas'])
            mask_sizes = areas / (512 * 512) * 100
            
            print(f"\n📐 掩膜大小分析:")
            print(f"  平均: {np.mean(mask_sizes):.1f}%")
            print(f"  中位数: {np.median(mask_sizes):.1f}%")
            print(f"  <10%: {np.sum(mask_sizes < 10)}个 ({np.sum(mask_sizes < 10)/len(mask_sizes)*100:.1f}%)")
            print(f"  10-20%: {np.sum((mask_sizes >= 10) & (mask_sizes < 20))}个")
            print(f"  20-50%: {np.sum((mask_sizes >= 20) & (mask_sizes < 50))}个")
            print(f"  >50%: {np.sum(mask_sizes >= 50)}个 ({np.sum(mask_sizes >= 50)/len(mask_sizes)*100:.1f}%)")
            
            # 关键改进指标
            small_masks_ratio = np.sum(mask_sizes < 20) / len(mask_sizes) * 100
            print(f"\n  关键指标:")
            print(f"    <20%掩膜比例: {small_masks_ratio:.1f}% (目标>50%)")
            if small_masks_ratio > 50:
                print(f"    ✅ 掩膜大小控制有效！")
            elif small_masks_ratio > 30:
                print(f"    ⚠️  有改善但需继续优化")
            else:
                print(f"    ❌ 掩膜大小控制不足")
    except:
        pass
    
    # 下一步决策
    print(f"\n{'='*80}")
    print(f"🎯 下一步行动:")
    
    if opt1_iou >= 6.0:
        print(f"  ✅ 优化成功！IoU达到{opt1_iou:.2f}%")
        print(f"  建议：长期训练（20000步）争取达到8-10%")
    elif opt1_iou >= 5.0:
        print(f"  ✅ 有改善！IoU {opt1_iou:.2f}%")
        print(f"  建议：继续优化循环2")
    elif opt1_iou > 3.86:
        print(f"  ⚠️  略有提升（+{opt1_iou-3.86:.2f}%）")
        print(f"  建议：调整超参数，进入优化循环2")
    else:
        print(f"  ⚠️  改进不明显")
        print(f"  建议：分析调试日志，重新设计奖励")
    
    print(f"{'='*80}\n")

