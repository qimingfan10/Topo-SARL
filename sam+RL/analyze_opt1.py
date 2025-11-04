#!/usr/bin/env python3
"""分析优化循环1的结果"""
import json
import collections
import numpy as np

# 加载数据
with open('logs/stage_b_opt1/final_metrics.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("优化循环1深度分析")
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
print(f"  >10%: {len([i for i in ious if i > 0.1])}个 ({len([i for i in ious if i > 0.1])/len(ious)*100:.1f}%)")
print(f"  <1%:  {len([i for i in ious if i < 0.01])}个 ({len([i for i in ious if i < 0.01])/len(ious)*100:.1f}%)")

# 4. 奖励-IoU关系
print(f"\n💰 奖励分析:")
rewards = data['episode_rewards']
print(f"  平均奖励: {np.mean(rewards):.4f}")
print(f"  奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")

# 5. 长episode vs 短episode对比
short_idx = [i for i, l in enumerate(lengths) if l <= 3]
long_idx = [i for i, l in enumerate(lengths) if l >= 5]

if short_idx and long_idx:
    short_ious = [ious[i] for i in short_idx]
    short_rewards = [rewards[i] for i in short_idx]
    long_ious = [ious[i] for i in long_idx]
    long_rewards = [rewards[i] for i in long_idx]
    
    print(f"\n🔍 长短Episode对比:")
    print(f"  短Episode (≤3步): {len(short_idx)}个")
    print(f"    平均IoU: {np.mean(short_ious)*100:.2f}%")
    print(f"    平均奖励: {np.mean(short_rewards):.4f}")
    print(f"  长Episode (≥5步): {len(long_idx)}个")
    print(f"    平均IoU: {np.mean(long_ious)*100:.2f}%")
    print(f"    平均奖励: {np.mean(long_rewards):.4f}")

# 6. 问题诊断
print(f"\n🔧 问题诊断:")
print(f"  1. 过早终止比例: {len(early_term)/len(lengths)*100:.1f}% (目标<30%)")
if len(early_term)/len(lengths) > 0.7:
    print(f"     ❌ 太高！智能体还是倾向快速终止")
elif len(early_term)/len(lengths) > 0.5:
    print(f"     ⚠️  偏高，需要继续优化")
else:
    print(f"     ✅ 可接受")

# 7. 根本原因分析
print(f"\n💡 根本原因推测:")
print(f"  1. 过早终止惩罚(-0.5)可能太小")
print(f"  2. min_steps_bonus(0.2)不足以激励探索")
print(f"  3. 智能体可能第1步就获得不错的IoU，倾向于快速收割")

# 8. 建议的改进
print(f"\n✨ 建议的改进方案:")
print(f"  方案1：激进惩罚")
print(f"    - 大幅增加过早终止惩罚：-0.5 → -2.0")
print(f"    - 增加min_steps_bonus：0.2 → 1.0")
print(f"    - 增加exploration_bonus：0.05 → 0.2")
print(f"  ")
print(f"  方案2：禁止过早终止")
print(f"    - 前5步禁止terminate动作（环境层面强制）")
print(f"    - 这样智能体被迫探索更多步")
print(f"  ")
print(f"  方案3：课程学习")
print(f"    - 开始时强制min_steps=10")
print(f"    - 随训练进度逐渐降低")

print("\n" + "="*80)

