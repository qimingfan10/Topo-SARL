#!/usr/bin/env python3
"""分析最终训练结果"""
import json
import collections
import numpy as np

# 加载数据
with open('logs/stage_b_final/metrics.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("🏁 最终训练结果分析（50000步）")
print("="*80)

# 1. Episode长度分布
lengths = data['episode_lengths']
len_counter = collections.Counter(lengths)

print("\n📏 Episode长度分布:")
for l in sorted(len_counter.keys())[:15]:
    count = len_counter[l]
    pct = count / len(lengths) * 100
    bar = "█" * int(pct / 2)
    print(f"  {l:2d}步: {count:4d}次 ({pct:5.1f}%) {bar}")

print(f"\n  总Episodes: {len(lengths)}个")
print(f"  平均长度: {np.mean(lengths):.2f}步")
print(f"  中位数: {np.median(lengths):.0f}步")

# 2. IoU统计
ious = data['episode_final_ious']
print(f"\n📊 IoU分布:")
print(f"  平均: {np.mean(ious)*100:.2f}%")
print(f"  中位数: {np.median(ious)*100:.2f}%")
print(f"  最大: {np.max(ious)*100:.2f}%")
print(f"  标准差: {np.std(ious)*100:.2f}%")
print(f"\n  >30%: {len([i for i in ious if i > 0.3])}个 ({len([i for i in ious if i > 0.3])/len(ious)*100:.1f}%)")
print(f"  >20%: {len([i for i in ious if i > 0.2])}个 ({len([i for i in ious if i > 0.2])/len(ious)*100:.1f}%)")
print(f"  >10%: {len([i for i in ious if i > 0.1])}个 ({len([i for i in ious if i > 0.1])/len(ious)*100:.1f}%)")

# 3. 三个版本对比
print(f"\n📈 完整对比:")
print(f"┌{'─'*30}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┐")
print(f"│ {'指标':<28} │ {'阶段A':<10} │ {'优化1':<10} │ {'优化2':<10} │ {'最终':<10} │")
print(f"├{'─'*30}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┤")
print(f"│ {'平均IoU':<28} │ {'1.08%':<10} │ {'4.64%':<10} │ {'18.35%':<10} │ {f'{np.mean(ious)*100:.2f}%':<10} │")
print(f"│ {'最佳IoU':<28} │ {'32.84%':<10} │ {'76.06%':<10} │ {'43.73%':<10} │ {f'{np.max(ious)*100:.2f}%':<10} │")
print(f"│ {'Episode长度':<28} │ {'8.75步':<10} │ {'2.98步':<10} │ {'6.93步':<10} │ {f'{np.mean(lengths):.2f}步':<10} │")
print(f"│ {'训练步数':<28} │ {'10000':<10} │ {'5000':<10} │ {'10000':<10} │ {'50000':<10} │")
print(f"└{'─'*30}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┘")

# 4. 与阶段A对比
stage_a_iou = 1.08
final_iou = np.mean(ious) * 100
print(f"\n🏆 最终成就:")
print(f"  阶段A (基准):     {stage_a_iou:.2f}%")
print(f"  阶段B (最终):     {final_iou:.2f}%")
print(f"  提升倍数:         {final_iou/stage_a_iou:.1f}x")
print(f"  绝对提升:         +{final_iou - stage_a_iou:.2f}%")

# 5. 训练稳定性分析
early_eps = 141
mid_eps = 566
late_eps = len(ious) - early_eps - mid_eps

early_ious = ious[:early_eps]
mid_ious = ious[early_eps:early_eps+mid_eps]
late_ious = ious[early_eps+mid_eps:]

print(f"\n📉 训练稳定性:")
print(f"  早期 (0-141):     {np.mean(early_ious)*100:.2f}%")
print(f"  中期 (142-707):   {np.mean(mid_ious)*100:.2f}%")
print(f"  后期 (708+):      {np.mean(late_ious)*100:.2f}%")
print(f"  变化:            {(np.mean(late_ious) - np.mean(early_ious))*100:+.2f}%")

# 6. 目标达成评估
print(f"\n✅ 目标达成情况:")
print(f"  ✓ 10% IoU目标:    {'✅ 达成' if final_iou >= 10 else '❌ 未达成'} ({final_iou:.2f}%)")
print(f"  ✓ Episode长度>5:  {'✅ 达成' if np.mean(lengths) > 5 else '❌ 未达成'} ({np.mean(lengths):.2f}步)")
print(f"  ✓ 优于阶段A:      {'✅ 达成' if final_iou > 1.08 else '❌ 未达成'} ({final_iou/stage_a_iou:.1f}x)")
print(f"  ✓ 过早终止<30%:   {'✅ 达成' if len([l for l in lengths if l<5])/len(lengths) < 0.3 else '❌ 未达成'} ({len([l for l in lengths if l<5])/len(lengths)*100:.1f}%)")

print(f"\n🎉 最终评价:")
if final_iou >= 18.0:
    print(f"  🌟🌟🌟 卓越成就！超额完成目标！")
    print(f"  - 平均IoU {final_iou:.2f}% (目标10%的 {final_iou/10:.1f}倍)")
    print(f"  - 相比阶段A提升 {final_iou/stage_a_iou:.1f}倍")
    print(f"  - 成果足以发表高质量论文")
elif final_iou >= 15.0:
    print(f"  ⭐⭐ 优秀结果！大幅超越目标！")
elif final_iou >= 10.0:
    print(f"  ⭐ 良好结果！达成10%目标！")
else:
    print(f"  ⚠️ 需要继续优化")

print("\n" + "="*80)

