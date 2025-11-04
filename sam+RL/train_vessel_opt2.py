#!/usr/bin/env python3
"""
血管优化循环2：暴力惩罚 + 负样本强制
- 掩膜大小惩罚增加10倍（-5.0）
- 强制负样本机制（掩膜>30%时70%概率转negative）
- 增加最小步数到10步
"""
import sys
import yaml
import numpy as np

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    config_path = '/home/ubuntu/sam+RL/config/stage_b_vessel_opt2.yaml'
    
    class Args:
        config = config_path
    
    print(f"\n{'='*80}")
    print(f"🔧 血管优化循环2（12000步）")
    print(f"{'='*80}")
    print(f"\n📊 上一轮性能（opt1）:")
    print(f"  平均IoU: 4.43%")
    print(f"  掩膜大小: 68.1% (严重过大！)")
    print(f"  <20%掩膜比例: 8.0% (目标>50%)")
    print(f"\n❌ Opt1失败原因:")
    print(f"  掩膜惩罚-0.5太弱 → 68%掩膜惩罚-0.24")
    print(f"  IoU奖励+0.20能抵消 → agent学会生成大掩膜")
    print(f"  假阳性率95%！")
    print(f"\n🎯 本轮改进（激进版）:")
    print(f"  1. 暴力惩罚: 掩膜惩罚 -0.5 → -5.0 (10倍)")
    print(f"     → 68%掩膜惩罚: -2.65 (能压制IoU奖励)")
    print(f"  2. 负样本强制: 掩膜>30%时70%概率转negative")
    print(f"  3. 增加最小步数: 7步 → 10步")
    print(f"  4. 降低IoU权重: 20 → 15 (减少方差)")
    print(f"  5. 增加探索: ent_coef 0.3 → 0.35")
    print(f"\n预期目标:")
    print(f"  • 平均IoU: 4.43% → 6-8%")
    print(f"  • 掩膜大小: 68.1% → 20-30%")
    print(f"  • <20%掩膜比例: 8% → 40%+")
    print(f"{'='*80}\n")
    
    # 运行训练
    print("开始训练...")
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"🏁 优化循环2完成")
    print(f"{'='*80}\n")
    
    opt2_iou = summary.get("avg_final_iou", 0) * 100
    opt2_best = summary.get("best_iou", 0) * 100
    opt2_len = summary.get("avg_episode_length", 0)
    
    print(f"结果对比:")
    print(f"┌{'─'*35}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*10}┐")
    print(f"│ {'指标':<33} │ {'基准':<10} │ {'Opt1':<10} │ {'Opt2':<10} │ {'变化':<8} │")
    print(f"├{'─'*35}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*10}┤")
    print(f"│ {'平均IoU':<33} │ {'3.86%':<10} │ {'4.43%':<10} │ {f'{opt2_iou:.2f}%':<10} │ {f'{opt2_iou-4.43:+.2f}%':<8} │")
    print(f"│ {'最佳IoU':<33} │ {'57.15%':<10} │ {'56.31%':<10} │ {f'{opt2_best:.2f}%':<10} │ {f'{opt2_best-56.31:+.2f}%':<8} │")
    print(f"│ {'Episode长度':<33} │ {'7.42步':<10} │ {'9.17步':<10} │ {f'{opt2_len:.2f}步':<10} │ {f'{opt2_len-9.17:+.2f}':<8} │")
    print(f"└{'─'*35}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*10}┘")
    
    # 详细分析
    import json
    metrics_file = 'logs/stage_b_vessel_opt2/final_metrics.json'
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # 分析掩膜大小
        if 'episode_final_areas' in data:
            areas = np.array(data['episode_final_areas'])
            mask_sizes = areas / (512 * 512) * 100
            
            print(f"\n📐 掩膜大小分析:")
            print(f"  平均: {np.mean(mask_sizes):.1f}% (opt1: 68.1%)")
            print(f"  中位数: {np.median(mask_sizes):.1f}% (opt1: 80.6%)")
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
            
            # 关键改进指标
            small_masks_ratio = (small + mid_small) / total * 100
            large_masks_ratio = large / total * 100
            
            print(f"\n  关键指标:")
            print(f"    <20%掩膜比例: {small_masks_ratio:.1f}% (opt1: 8.0%, 目标>40%)")
            print(f"    >50%掩膜比例: {large_masks_ratio:.1f}% (opt1: 72.0%, 目标<20%)")
            
            improvement_score = 0
            if small_masks_ratio > 40:
                print(f"    ✅ 小掩膜比例达标！")
                improvement_score += 2
            elif small_masks_ratio > 20:
                print(f"    ⚠️  有改善但未达标")
                improvement_score += 1
            else:
                print(f"    ❌ 掩膜大小控制仍不足")
            
            if large_masks_ratio < 20:
                print(f"    ✅ 大掩膜比例显著降低！")
                improvement_score += 2
            elif large_masks_ratio < 50:
                print(f"    ⚠️  有降低但仍较高")
                improvement_score += 1
            else:
                print(f"    ❌ 大掩膜问题未解决")
        
        # 分析负样本强制效果
        if 'forced_negative_count' in data:
            forced = data['forced_negative_count']
            print(f"\n  负样本强制统计:")
            print(f"    触发次数: {forced}次")
    except Exception as e:
        print(f"\n⚠️  无法加载详细指标: {e}")
    
    # 评估改进效果
    print(f"\n📊 改进评估:")
    
    iou_improved = opt2_iou > 4.43
    iou_target_met = opt2_iou >= 6.0
    
    if iou_target_met:
        print(f"  🎉 IoU达标！{opt2_iou:.2f}% ≥ 6.0%")
    elif iou_improved:
        print(f"  ✅ IoU提升 {opt2_iou-4.43:.2f}%")
    else:
        print(f"  ⚠️  IoU未提升")
    
    # 下一步决策
    print(f"\n{'='*80}")
    print(f"🎯 下一步行动:")
    
    if opt2_iou >= 8.0:
        print(f"  🎉 优化非常成功！IoU {opt2_iou:.2f}%")
        print(f"  建议：长期训练（30000步）冲击10%目标")
    elif opt2_iou >= 6.0:
        print(f"  ✅ 优化成功！IoU {opt2_iou:.2f}%")
        print(f"  建议：进行长期训练或尝试更激进优化")
    elif opt2_iou > 4.43:
        print(f"  ⚠️  有改善（+{opt2_iou-4.43:.2f}%）但未达6%目标")
        print(f"  建议：进入优化循环3，尝试更激进策略")
    else:
        print(f"  ❌ 改进不明显或退步")
        print(f"  建议：重新分析问题，可能需要改变策略")
    
    print(f"{'='*80}\n")

