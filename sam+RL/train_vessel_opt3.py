#!/usr/bin/env python3
"""
血管优化循环3：Precision奖励（精确度优先）
- 放弃暴力惩罚
- 使用precision作为主要奖励信号
- precision = IoU / mask_coverage（自然鼓励小而精准的掩膜）
"""
import sys
import yaml
import numpy as np

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    config_path = '/home/ubuntu/sam+RL/config/stage_b_vessel_opt3.yaml'
    
    class Args:
        config = config_path
    
    print(f"\n{'='*80}")
    print(f"🔧 血管优化循环3（12000步）- Precision奖励")
    print(f"{'='*80}")
    print(f"\n📊 前两轮回顾:")
    print(f"  Opt1: IoU 4.43%, 掩膜68.1%, 平均奖励+0.33")
    print(f"  Opt2: IoU 4.26%, 掩膜58.2%, 平均奖励-22.2 ❌ 失败")
    print(f"\n❌ 失败原因:")
    print(f"  暴力惩罚(-5.0)导致奖励崩溃")
    print(f"  掩膜变小不等于IoU提升")
    print(f"  根本矛盾：小掩膜如果不精准，IoU反而更低")
    print(f"\n🎯 Opt3核心创新：Precision奖励")
    print(f"  precision = IoU / mask_coverage")
    print(f"  ")
    print(f"  例子1: IoU=4%, mask=60% → precision=6.7%  → 低奖励")
    print(f"  例子2: IoU=4%, mask=10% → precision=40%   → 高奖励（6倍！）")
    print(f"  ")
    print(f"  原理：自然鼓励'小而精准'的掩膜")
    print(f"        不鼓励'小但不精准'的掩膜")
    print(f"\n本轮改进:")
    print(f"  1. Precision权重×30（主要奖励信号）")
    print(f"  2. IoU权重降低（10而非15）")
    print(f"  3. 温和大小引导（>80%才惩罚-1.0）")
    print(f"  4. 关闭强制负样本（让agent自己学）")
    print(f"\n预期目标:")
    print(f"  • 平均IoU: 4.43% → 5.5%+")
    print(f"  • 平均precision: 15%+")
    print(f"  • <20%掩膜比例: 20%+")
    print(f"  • 平均奖励: 正值（不崩溃）")
    print(f"{'='*80}\n")
    
    # 运行训练
    print("开始训练...")
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"🏁 优化循环3完成")
    print(f"{'='*80}\n")
    
    opt3_iou = summary.get("avg_final_iou", 0) * 100
    opt3_best = summary.get("best_iou", 0) * 100
    opt3_len = summary.get("avg_episode_length", 0)
    opt3_reward = summary.get("avg_episode_reward", 0)
    
    print(f"结果对比:")
    print(f"┌{'─'*35}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*10}┐")
    print(f"│ {'指标':<33} │ {'基准':<10} │ {'Opt1':<10} │ {'Opt2':<10} │ {'Opt3':<10} │ {'vs Opt1':<8} │")
    print(f"├{'─'*35}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*10}┤")
    print(f"│ {'平均IoU':<33} │ {'3.86%':<10} │ {'4.43%':<10} │ {'4.26%':<10} │ {f'{opt3_iou:.2f}%':<10} │ {f'{opt3_iou-4.43:+.2f}%':<8} │")
    print(f"│ {'最佳IoU':<33} │ {'57.15%':<10} │ {'56.31%':<10} │ {'52.88%':<10} │ {f'{opt3_best:.2f}%':<10} │ {f'{opt3_best-56.31:+.2f}%':<8} │")
    print(f"│ {'Episode长度':<33} │ {'7.42步':<10} │ {'9.17步':<10} │ {'11.87步':<10} │ {f'{opt3_len:.2f}步':<10} │ {f'{opt3_len-9.17:+.2f}':<8} │")
    print(f"│ {'平均奖励':<33} │ {'+2.17':<10} │ {'+0.33':<10} │ {'-22.2':<10} │ {f'{opt3_reward:+.2f}':<10} │ {f'{opt3_reward-0.33:+.2f}':<8} │")
    print(f"└{'─'*35}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*10}┘")
    
    # 详细分析
    import json
    metrics_file = 'logs/stage_b_vessel_opt3/final_metrics.json'
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # 分析掩膜大小和precision
        if 'episode_final_areas' in data:
            areas = np.array(data['episode_final_areas'])
            ious = np.array(data['episode_final_ious'])
            mask_sizes = areas / (512 * 512) * 100
            precisions = (ious * 100) / (mask_sizes + 1e-8)
            
            print(f"\n📐 掩膜大小分析:")
            print(f"  平均: {np.mean(mask_sizes):.1f}% (opt1: 68.1%, opt2: 58.2%)")
            print(f"  中位数: {np.median(mask_sizes):.1f}% (opt1: 80.6%, opt2: 62.4%)")
            
            print(f"\n  分布:")
            small = np.sum(mask_sizes < 10)
            mid_small = np.sum((mask_sizes >= 10) & (mask_sizes < 20))
            mid = np.sum((mask_sizes >= 20) & (mask_sizes < 50))
            large = np.sum(mask_sizes >= 50)
            total = len(mask_sizes)
            
            print(f"    <10%:   {small:3d}个 ({small/total*100:5.1f}%) (opt1: 3.8%)")
            print(f"    10-20%: {mid_small:3d}个 ({mid_small/total*100:5.1f}%)")
            print(f"    20-50%: {mid:3d}个 ({mid/total*100:5.1f}%)")
            print(f"    >50%:   {large:3d}个 ({large/total*100:5.1f}%) (opt1: 72.0%)")
            
            small_masks_ratio = (small + mid_small) / total * 100
            large_masks_ratio = large / total * 100
            
            print(f"\n  关键指标:")
            print(f"    <20%掩膜比例: {small_masks_ratio:.1f}% (opt1: 8.0%, 目标>20%)")
            print(f"    >50%掩膜比例: {large_masks_ratio:.1f}% (opt1: 72.0%, 目标<20%)")
            
            # Precision分析（新增）
            print(f"\n🎯 Precision分析（关键指标）:")
            print(f"  平均precision: {np.mean(precisions):.1f}%")
            print(f"  中位precision: {np.median(precisions):.1f}%")
            print(f"  最佳precision: {np.max(precisions):.1f}%")
            
            high_prec = np.sum(precisions > 20)
            mid_prec = np.sum((precisions >= 10) & (precisions <= 20))
            low_prec = np.sum(precisions < 10)
            
            print(f"\n  Precision分布:")
            print(f"    >20%:   {high_prec:3d}个 ({high_prec/total*100:5.1f}%) ← 目标")
            print(f"    10-20%: {mid_prec:3d}个 ({mid_prec/total*100:5.1f}%)")
            print(f"    <10%:   {low_prec:3d}个 ({low_prec/total*100:5.1f}%)")
            
            # 相关性分析
            print(f"\n  Precision vs 掩膜大小相关性:")
            small_mask_prec = precisions[mask_sizes < 20]
            large_mask_prec = precisions[mask_sizes >= 50]
            if len(small_mask_prec) > 0:
                print(f"    小掩膜(<20%)的平均precision: {np.mean(small_mask_prec):.1f}%")
            if len(large_mask_prec) > 0:
                print(f"    大掩膜(≥50%)的平均precision: {np.mean(large_mask_prec):.1f}%")
            
            # 评估改进
            improvement_score = 0
            
            if opt3_iou > 4.43:
                print(f"\n  ✅ IoU提升 {opt3_iou-4.43:.2f}%")
                improvement_score += 2
            
            if small_masks_ratio > 20:
                print(f"  ✅ 小掩膜比例显著提升")
                improvement_score += 2
            elif small_masks_ratio > 8:
                print(f"  ⚠️  小掩膜比例有改善")
                improvement_score += 1
            
            if np.mean(precisions) > 15:
                print(f"  ✅ Precision达标（{np.mean(precisions):.1f}% > 15%）")
                improvement_score += 2
            elif np.mean(precisions) > 10:
                print(f"  ⚠️  Precision有改善（{np.mean(precisions):.1f}%）")
                improvement_score += 1
            
            if opt3_reward > 0:
                print(f"  ✅ 奖励正常（未崩溃）")
                improvement_score += 1
    except Exception as e:
        print(f"\n⚠️  无法加载详细指标: {e}")
        improvement_score = 0
    
    # 总体评估
    print(f"\n{'='*80}")
    print(f"🎯 总体评估:")
    
    if opt3_iou >= 6.0:
        print(f"  🎉 优化非常成功！IoU {opt3_iou:.2f}% ≥ 6.0%")
        print(f"  建议：长期训练（30000步）冲击8-10%目标")
    elif opt3_iou >= 5.5:
        print(f"  ✅ 优化成功！IoU {opt3_iou:.2f}% ≥ 5.5%")
        print(f"  建议：进行长期训练或继续微调")
    elif opt3_iou > 4.43:
        print(f"  ⚠️  有改善（+{opt3_iou-4.43:.2f}%）")
        print(f"  建议：分析precision机制效果，可能需要继续优化")
    else:
        print(f"  ❌ IoU未提升")
        print(f"  建议：重新审视问题，可能需要改变模型或任务定义")
    
    print(f"\n下一步行动:")
    if opt3_iou >= 5.5 and opt3_reward > 0:
        print(f"  → 进入长期训练阶段（20000-30000步）")
    elif opt3_iou > 4.43:
        print(f"  → 进入优化循环4，微调precision权重")
    else:
        print(f"  → 深度分析precision机制，考虑多目标优化")
    
    print(f"{'='*80}\n")

