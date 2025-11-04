"""
测试数据加载和奖励计算
"""
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.append('/home/ubuntu/sam+RL')

from utils.data_loader import VesselDataset
from rewards.reward_functions import RewardCalculator, compute_vesselness

def main():
    print("=" * 60)
    print("数据和奖励测试")
    print("=" * 60)
    print()
    
    # 1. 测试数据加载
    print("1. 测试数据加载...")
    dataset = VesselDataset(
        image_dir="/home/ubuntu/Segment_DATA/orgin_pic",
        mask_dir="/home/ubuntu/Segment_DATA/lab_pic",
        image_size=(512, 512)
    )
    print(f"   数据集大小: {len(dataset)} 个样本")
    print()
    
    # 2. 获取一个样本
    print("2. 获取第一个样本...")
    sample = dataset[0]
    image = sample['image']
    mask = sample.get('mask', None)
    name = sample['name']
    
    print(f"   样本名称: {name}")
    print(f"   图像形状: {image.shape}, 类型: {image.dtype}, 范围: [{image.min()}, {image.max()}]")
    if mask is not None:
        print(f"   掩膜形状: {mask.shape}, 类型: {mask.dtype}, True像素: {mask.sum()}")
    else:
        print(f"   ⚠️  警告：未找到掩膜！")
    print()
    
    # 3. 可视化
    print("3. 保存可视化...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title(f'原始图像\n{name}')
    axes[0].axis('off')
    
    # 掩膜
    if mask is not None:
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'标注掩膜\n像素数: {mask.sum()}')
        axes[1].axis('off')
    
        # 叠加
        overlay = image.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
    
    plt.tight_layout()
    output_path = '/home/ubuntu/sam+RL/data_test_sample.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 保存到: {output_path}")
    plt.close()
    print()
    
    # 4. 测试 vesselness 计算
    print("4. 测试 vesselness 计算...")
    if mask is not None and mask.sum() > 0:
        # 计算整个图像的 vesselness
        vesselness_full = compute_vesselness(image, mask=None)
        print(f"   全图 vesselness: {vesselness_full:.6f}")
        
        # 计算掩膜内的 vesselness
        vesselness_mask = compute_vesselness(image, mask=mask)
        print(f"   掩膜内 vesselness: {vesselness_mask:.6f}")
    print()
    
    # 5. 测试奖励计算
    print("5. 测试奖励计算...")
    reward_config = {
        'use_gt': True,
        'iou_weight': 2.0,
        'cldice_weight': 0.5,
        'topology_weight': 0.1,
        'vesselness_weight': 1.0,
        'smoothness_weight': 0.2,
        'action_cost': -0.002,
        'false_positive_penalty': -0.5
    }
    
    reward_calc = RewardCalculator(reward_config)
    
    if mask is not None:
        # 创建一个测试预测（与GT相同）
        pred_mask = mask.copy()
        
        # 计算奖励
        reward_dict = reward_calc.compute_reward(
            pred_mask=pred_mask,
            gt_mask=mask,
            image=image,
            action_type='select'
        )
        
        print("   完美匹配的奖励（pred == gt）:")
        for key, value in reward_dict.items():
            print(f"     {key}: {value:.4f}")
        print()
        
        # 测试部分匹配
        pred_mask_partial = mask.copy()
        # 只保留50%的像素
        indices = np.where(mask)
        keep_indices = np.random.choice(len(indices[0]), size=len(indices[0])//2, replace=False)
        pred_mask_partial = np.zeros_like(mask)
        pred_mask_partial[indices[0][keep_indices], indices[1][keep_indices]] = True
        
        reward_dict_partial = reward_calc.compute_reward(
            pred_mask=pred_mask_partial,
            gt_mask=mask,
            image=image,
            action_type='select'
        )
        
        print("   部分匹配的奖励（50%重叠）:")
        for key, value in reward_dict_partial.items():
            print(f"     {key}: {value:.4f}")
    
    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

