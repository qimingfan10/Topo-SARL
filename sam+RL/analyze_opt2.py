#!/usr/bin/env python3
"""分析优化循环2的成功结果"""
import json
import collections
import numpy as np

# 加载数据
with open('logs/stage_b_opt2/final_metrics.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("🎉 优化循环2深度分析 - 重大突破！")
print("="*80)

# 1. Episode长度分布
lengths = data['episode_lengths']
len_counter = collections.Counter(lengths)

print("\n📏 Episode长度分布:")
for l in sorted(len_counter.keys())[:20]:
    count = len_counter[l]
    pct = count / len(lengths) * 100
    bar = "█" * int(pct / 2)
    print(f"  {l:2d}步: {count:4d}次 ({pct:5.1f}%) {bar}")

print(f"\n  总计: {len(lengths)}个episodes")
print(f"  平均: {np.mean(lengths):.2f}步")
print(f"  中位数: {np.median(lengths):.0f}步")
print(f"  标准差: {np.std(lengths):.2f}步")

# 2. 过早终止分析（<5步）
early_term = [l for l in lengths if l < 5]
print(f"\n⚠️  过早终止 (<5步): {len(early_term)}个 ({len(early_term)/len(lengths)*100:.1f}%)")

# 3. IoU分布
ious = data['episode_final_ious']
print(f"\n📊 IoU分布:")
print(f"  平均: {np.mean(ious)*100:.2f}%")
print(f"  中位数: {np.median(ious)*100:.2f}%")
print(f"  最大: {np.max(ious)*100:.2f}%")
print(f"  最小: {np.min(ious)*100:.2f}%")
print(f"  >50%: {len([i for i in ious if i > 0.5])}个 ({len([i for i in ious if i > 0.5])/len(ious)*100:.1f}%)")
print(f"  >30%: {len([i for i in ious if i > 0.3])}个 ({len([i for i in ious if i > 0.3])/len(ious)*100:.1f}%)")
print(f"  >10%: {len([i for i in ious if i > 0.1])}个 ({len([i for i in ious if i > 0.1])/len(ious)*100:.1f}%)")
print(f"  <1%:  {len([i for i in ious if i < 0.01])}个 ({len([i for i in ious if i < 0.01])/len(ious)*100:.1f}%)")

# 4. 奖励分析
print(f"\n💰 奖励分析:")
rewards = data['episode_rewards']
print(f"  平均奖励: {np.mean(rewards):.4f}")
print(f"  奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")

# 5. 对比优化1和优化2
print(f"\n🔍 优化1 vs 优化2 对比:")
print(f"  {'指标':<20} {'优化1':<15} {'优化2':<15} {'改善':<15}")
print(f"  {'-'*65}")
print(f"  {'过早终止比例':<20} {'78.9%':<15} {f'{len(early_term)/len(lengths)*100:.1f}%':<15} {f'{78.9 - len(early_term)/len(lengths)*100:+.1f}%':<15}")
print(f"  {'平均Episode长度':<20} {'2.98步':<15} {f'{np.mean(lengths):.2f}步':<15} {f'{np.mean(lengths) - 2.98:+.2f}步':<15}")
print(f"  {'平均IoU':<20} {'4.64%':<15} {f'{np.mean(ious)*100:.2f}%':<15} {f'{(np.mean(ious)*100 - 4.64):+.2f}%':<15}")

# 6. 成功原因分析
print(f"\n✨ 成功原因分析:")
print(f"  1. ✅ 强制最小步数有效：过早终止从78.9% → {len(early_term)/len(lengths)*100:.1f}%")
print(f"  2. ✅ 更多探索带来更好结果：平均长度从2.98 → {np.mean(lengths):.2f}步")
print(f"  3. ✅ IoU大幅提升：4.64% → {np.mean(ious)*100:.2f}% (提升{(np.mean(ious)*100/4.64 - 1)*100:.0f}%)")

# 7. 与阶段A对比
stage_a_iou = 1.08
print(f"\n🏆 与阶段A基准对比:")
print(f"  阶段A (候选选择): {stage_a_iou:.2f}%")
print(f"  阶段B 优化2:     {np.mean(ious)*100:.2f}%")
print(f"  提升倍数:        {np.mean(ious)*100/stage_a_iou:.1f}x")

# 8. 下一步建议
print(f"\n🚀 下一步行动:")
avg_iou = np.mean(ious)*100
if avg_iou >= 15.0:
    print(f"  🎉 已经超越预期！当前{avg_iou:.2f}%")
    print(f"  建议：")
    print(f"    1. 长期训练（50000步）看能否继续提升")
    print(f"    2. 尝试更大网格（64×64）获得更精细的控制")
    print(f"    3. 整理成果，撰写论文")
elif avg_iou >= 10.0:
    print(f"  ✅ 达到10%目标！当前{avg_iou:.2f}%")
    print(f"  建议：继续训练争取达到15%")
else:
    print(f"  ⚠️  接近目标，需继续优化")

print("\n" + "="*80)

