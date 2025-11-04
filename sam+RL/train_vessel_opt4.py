#!/usr/bin/env python3
"""
血管优化循环4：大幅增强Precision权重
- Precision权重: 30 → 100 (3倍)
- Delta IoU权重: 10 → 5 (减半)
- 目标：让precision主导策略学习
"""
import sys
import yaml
import numpy as np

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    config_path = '/home/ubuntu/sam+RL/config/stage_b_vessel_opt4.yaml'
    
    class Args:
        config = config_path
    
    print(f"\n{'='*80}")
    print(f"🔧 血管优化循环4（15000步）- 大幅增强Precision")
    print(f"{'='*80}")
    print(f"\n📊 Opt3回顾:")
    print(f"  平均IoU: 4.75%")
    print(f"  掩膜大小: 68.5% （未改善❌）")
    print(f"  平均奖励: +243.23 （恢复正常✅）")
    print(f"  <20%掩膜: 6.8% （目标>20%）")
    print(f"\n❌ Opt3问题:")
    print(f"  Precision奖励有效，但掩膜大小未改善")
    print(f"  原因：Delta IoU权重(×10)仍然太强")
    print(f"  Agent害怕尝试小掩膜（怕IoU下降）")
    print(f"\n🎯 Opt4核心改进:")
    print(f"  ")
    print(f"  策略：大幅增强Precision，大幅降低Delta IoU")
    print(f"  ")
    print(f"  1. Precision权重: 30 → 100 (3倍增强)")
    print(f"  2. Delta IoU权重: 10 → 5 (减半)")
    print(f"  ")
    print(f"  效果预测：")
    print(f"    大掩膜(68%, IoU=4%): precision=5.9% → R=590")
    print(f"    小掩膜(20%, IoU=4%): precision=20%  → R=2000 (3.4倍)")
    print(f"  ")
    print(f"    即使IoU下降1%，小掩膜仍高奖励！")
    print(f"      小掩膜: 2000 - 5 = 1995")
    print(f"      大掩膜: 590")
    print(f"      差距: 3.4倍")
    print(f"\n预期目标:")
    print(f"  • 平均IoU: 4.75% → 5.5-6.0%")
    print(f"  • <20%掩膜比例: 6.8% → 15-25%")
    print(f"  • 平均掩膜: 68.5% → 40-50%")
    print(f"  • Precision主导策略")
    print(f"{'='*80}\n")
    
    # 运行训练
    print("开始训练...")
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"🏁 优化循环4完成")
    print(f"{'='*80}\n")
    
    opt4_iou = summary.get("avg_final_iou", 0) * 100
    opt4_best = summary.get("best_iou", 0) * 100
    opt4_len = summary.get("avg_episode_length", 0)
    opt4_reward = summary.get("avg_episode_reward", 0)
    
    print(f"结果对比:")
    print(f"┌{'─'*30}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┐")
    print(f"│ {'指标':<28} │ {'Opt1':<8} │ {'Opt2':<8} │ {'Opt3':<8} │ {'Opt4':<8} │ {'vs Opt3':<8} │")
    print(f"├{'─'*30}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┤")
    print(f"│ {'平均IoU':<28} │ {'4.43%':<8} │ {'4.26%':<8} │ {'4.75%':<8} │ {f'{opt4_iou:.2f}%':<8} │ {f'{opt4_iou-4.75:+.2f}%':<8} │")
    print(f"│ {'最佳IoU':<28} │ {'56.31%':<8} │ {'52.88%':<8} │ {'44.46%':<8} │ {f'{opt4_best:.2f}%':<8} │ {f'{opt4_best-44.46:+.2f}%':<8} │")
    print(f"│ {'Episode长度':<28} │ {'9.17步':<8} │ {'11.87步':<8} │ {'10.73步':<8} │ {f'{opt4_len:.1f}步':<8} │ {f'{opt4_len-10.73:+.2f}':<8} │")
    print(f"│ {'平均奖励':<28} │ {'+0.33':<8} │ {'-22.2':<8} │ {'+243':<8} │ {f'{opt4_reward:+.0f}':<8} │ {f'{opt4_reward-243:+.0f}':<8} │")
    print(f"└{'─'*30}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┘")
    
    # 详细分析
    import json
    metrics_file = 'logs/stage_b_vessel_opt4/final_metrics.json'
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        if 'episode_final_areas' in data:
            areas = np.array(data['episode_final_areas'])
            ious = np.array(data['episode_final_ious'])
            mask_sizes_pct = areas / (512 * 512) * 100
            mask_sizes_ratio = areas / (512 * 512)
            precisions_pct = np.divide(ious * 100, mask_sizes_pct, where=mask_sizes_pct>0, out=np.zeros_like(ious))
            
            print(f"\n📐 掩膜大小分析:")
            print(f"  平均: {np.mean(mask_sizes_pct):.1f}% (opt3: 68.5%, 目标<50%)")
            print(f"  中位数: {np.median(mask_sizes_pct):.1f}% (opt3: 80.7%)")
            
            small = np.sum(mask_sizes_pct < 10)
            mid_small = np.sum((mask_sizes_pct >= 10) & (mask_sizes_pct < 20))
            mid = np.sum((mask_sizes_pct >= 20) & (mask_sizes_pct < 50))
            large = np.sum(mask_sizes_pct >= 50)
            total = len(mask_sizes_pct)
            
            print(f"\n  分布:")
            print(f"    <10%:   {small:3d}个 ({small/total*100:5.1f}%) (opt3: 2.8%)")
            print(f"    10-20%: {mid_small:3d}个 ({mid_small/total*100:5.1f}%) (opt3: 4.1%)")
            print(f"    20-50%: {mid:3d}个 ({mid/total*100:5.1f}%) (opt3: 20.7%)")
            print(f"    >50%:   {large:3d}个 ({large/total*100:5.1f}%) (opt3: 72.5%)")
            
            small_masks_ratio = (small + mid_small) / total * 100
            large_masks_ratio = large / total * 100
            
            print(f"\n  关键指标:")
            print(f"    <20%掩膜比例: {small_masks_ratio:.1f}% (opt3: 6.8%, 目标>15%)")
            print(f"    >50%掩膜比例: {large_masks_ratio:.1f}% (opt3: 72.5%, 目标<50%)")
            
            # Precision分析
            print(f"\n🎯 Precision分析:")
            print(f"  平均precision: {np.mean(precisions_pct):.1f}%")
            print(f"  中位precision: {np.median(precisions_pct):.1f}%")
            print(f"  最佳precision: {np.max(precisions_pct):.1f}%")
            
            high_prec = np.sum(precisions_pct > 20)
            mid_prec = np.sum((precisions_pct >= 10) & (precisions_pct <= 20))
            low_prec = np.sum(precisions_pct < 10)
            
            print(f"\n  Precision分布:")
            print(f"    >20%:   {high_prec:3d}个 ({high_prec/total*100:5.1f}%)")
            print(f"    10-20%: {mid_prec:3d}个 ({mid_prec/total*100:5.1f}%)")
            print(f"    <10%:   {low_prec:3d}个 ({low_prec/total*100:5.1f}%)")
            
            # 改进评估
            print(f"\n📊 改进评估:")
            
            improvements = []
            
            if opt4_iou >= 6.0:
                improvements.append(f"  🎉 IoU突破6%！({opt4_iou:.2f}%)")
            elif opt4_iou > 4.75:
                improvements.append(f"  ✅ IoU提升 {opt4_iou-4.75:.2f}%")
            
            if small_masks_ratio >= 15:
                improvements.append(f"  ✅ 小掩膜比例达标 ({small_masks_ratio:.1f}% ≥ 15%)")
            elif small_masks_ratio > 6.8:
                improvements.append(f"  ⚠️  小掩膜比例改善 ({small_masks_ratio:.1f}%)")
            
            if np.mean(mask_sizes_pct) < 50:
                improvements.append(f"  ✅ 平均掩膜显著降低 ({np.mean(mask_sizes_pct):.1f}% < 50%)")
            elif np.mean(mask_sizes_pct) < 60:
                improvements.append(f"  ⚠️  平均掩膜有所降低 ({np.mean(mask_sizes_pct):.1f}%)")
            
            if improvements:
                for imp in improvements:
                    print(imp)
            else:
                print(f"  ⚠️  改进不明显")
    except Exception as e:
        print(f"\n⚠️  无法加载详细指标: {e}")
    
    # 总体评估和下一步
    print(f"\n{'='*80}")
    print(f"🎯 总体评估和下一步:")
    
    if opt4_iou >= 6.5:
        print(f"  🎉 优化非常成功！IoU {opt4_iou:.2f}% ≥ 6.5%")
        print(f"  → 进入长期训练（30000步）冲击8-10%")
    elif opt4_iou >= 5.5:
        print(f"  ✅ 优化成功！IoU {opt4_iou:.2f}% ≥ 5.5%")
        print(f"  → 长期训练或继续微调达到6%+")
    elif opt4_iou > 4.75:
        print(f"  ⚠️  有改善（+{opt4_iou-4.75:.2f}%）")
        print(f"  → 继续优化循环5，可能需要更激进策略")
    else:
        print(f"  ❌ 未改善")
        print(f"  → 重新审视问题，可能需要改变方法")
    
    print(f"{'='*80}\n")

