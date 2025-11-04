#!/usr/bin/env python3
"""验证新数据集的完整性"""
import sys
sys.path.insert(0, '/home/ubuntu/sam+RL')

from utils.data_loader import VesselDataset
import numpy as np

print("\n" + "="*80)
print("数据集验证")
print("="*80)

# 加载数据集
dataset = VesselDataset(
    image_dir='/home/ubuntu/Segment_DATA/orgin_pic',
    mask_dir='/home/ubuntu/Segment_DATA/lab_pic',
    image_size=(512, 512)
)

print(f"\n✓ 数据集加载成功")
print(f"  图像数量: {len(dataset)}")

# 检查前5个样本
print(f"\n检查前5个样本:")
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    image = sample['image']
    mask = sample.get('mask', None)
    name = sample['name']
    
    print(f"\n样本 {i+1}:")
    print(f"  名称: {name}")
    print(f"  图像尺寸: {image.shape}")
    if mask is not None:
        print(f"  掩膜尺寸: {mask.shape}")
        print(f"  掩膜覆盖率: {mask.sum() / mask.size * 100:.2f}%")
        print(f"  掩膜像素数: {mask.sum()}")
    else:
        print(f"  ⚠️  无掩膜")

# 统计所有样本的掩膜覆盖率
print(f"\n" + "="*80)
print("整体统计:")
print("="*80)

coverage_rates = []
has_mask_count = 0
no_mask_count = 0

for i in range(len(dataset)):
    sample = dataset[i]
    mask = sample.get('mask', None)
    if mask is not None:
        coverage = mask.sum() / mask.size * 100
        coverage_rates.append(coverage)
        has_mask_count += 1
    else:
        no_mask_count += 1

print(f"\n有掩膜的样本: {has_mask_count}")
print(f"无掩膜的样本: {no_mask_count}")

if coverage_rates:
    print(f"\n掩膜覆盖率统计:")
    print(f"  平均: {np.mean(coverage_rates):.2f}%")
    print(f"  中位数: {np.median(coverage_rates):.2f}%")
    print(f"  最大: {np.max(coverage_rates):.2f}%")
    print(f"  最小: {np.min(coverage_rates):.2f}%")

print(f"\n" + "="*80)
print("✓ 数据集验证完成")
print("="*80)

