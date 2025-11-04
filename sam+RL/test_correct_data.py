"""
测试正确的数据集加载
验证220个原始造影图像和标注掩膜
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/sam+RL')

from utils.data_loader import VesselDataset
from rewards.reward_functions import compute_vesselness, compute_iou, RewardCalculator

def test_data_loading():
    """测试数据加载"""
    print("=" * 80)
    print("测试数据加载")
    print("=" * 80)
    
    # 创建数据集
    dataset = VesselDataset(
        image_dir="/home/ubuntu/Segment_DATA/orgin_pic",
        mask_dir="/home/ubuntu/Segment_DATA/lab_pic",
        image_size=(512, 512)
    )
    
    print(f"\n✓ 数据集加载成功")
    print(f"  - 图像数量: {len(dataset)}")
    print(f"  - 图像目录: /home/ubuntu/Segment_DATA/orgin_pic")
    print(f"  - 掩膜目录: /home/ubuntu/Segment_DATA/lab_pic")
    
    # 测试加载几个样本
    print("\n" + "=" * 80)
    print("测试样本加载")
    print("=" * 80)
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        mask = sample.get('mask', None)
        name = sample['name']
        
        print(f"\n样本 {i+1}:")
        print(f"  - 名称: {name}")
        print(f"  - 图像形状: {image.shape}, 类型: {image.dtype}")
        print(f"  - 图像值范围: [{image.min()}, {image.max()}]")
        print(f"  - 唯一值数量: {len(np.unique(image))}")
        
        if mask is not None:
            print(f"  - 掩膜形状: {mask.shape}, 类型: {mask.dtype}")
            print(f"  - 掩膜面积: {mask.sum()} 像素 ({mask.sum()/(512*512)*100:.1f}%)")
            print(f"  - 掩膜唯一值: {np.unique(mask)}")
        else:
            print(f"  - ⚠️ 没有找到对应的掩膜")
    
    return dataset

def test_vesselness(dataset):
    """测试vesselness计算"""
    print("\n" + "=" * 80)
    print("测试Vesselness计算")
    print("=" * 80)
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        mask = sample.get('mask', None)
        
        # 转灰度图
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        print(f"\n样本 {i+1}:")
        print(f"  - 灰度图形状: {image_gray.shape}")
        print(f"  - 灰度图值范围: [{image_gray.min()}, {image_gray.max()}]")
        
        # 计算vesselness
        vesselness_score = compute_vesselness(image, mask)
        print(f"  - Vesselness得分: {vesselness_score:.6f}")

def test_reward_calculation(dataset):
    """测试奖励计算"""
    print("\n" + "=" * 80)
    print("测试奖励计算")
    print("=" * 80)
    
    # 创建奖励计算器
    config = {
        'use_gt': True,
        'iou_weight': 2.0,
        'cldice_weight': 0.5,
        'topology_weight': 0.1,
        'vesselness_weight': 1.0,
        'smoothness_weight': 0.2,
        'action_cost': -0.002,
        'false_positive_penalty': -0.5
    }
    
    reward_calc = RewardCalculator(config)
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        gt_mask = sample.get('mask', None)
        
        if gt_mask is None:
            print(f"\n样本 {i+1}: 跳过（无标注）")
            continue
        
        # 创建一个模拟的预测掩膜（随机选择一部分区域）
        pred_mask = np.zeros_like(gt_mask)
        # 随机选择80%的真实区域
        gt_indices = np.where(gt_mask)
        if len(gt_indices[0]) > 0:
            num_select = int(len(gt_indices[0]) * 0.8)
            select_indices = np.random.choice(len(gt_indices[0]), num_select, replace=False)
            pred_mask[gt_indices[0][select_indices], gt_indices[1][select_indices]] = True
        
        # 计算奖励
        rewards = reward_calc.compute_reward(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            image=image,
            prev_mask=None,
            action_type='select'
        )
        
        # 计算IoU
        iou = compute_iou(pred_mask, gt_mask)
        
        print(f"\n样本 {i+1}:")
        print(f"  - GT掩膜面积: {gt_mask.sum()} ({gt_mask.sum()/(512*512)*100:.1f}%)")
        print(f"  - 预测掩膜面积: {pred_mask.sum()} ({pred_mask.sum()/(512*512)*100:.1f}%)")
        print(f"  - IoU: {iou:.4f}")
        print(f"  - 奖励详情: {rewards}")
        print(f"  - 总奖励: {rewards['total']:.4f}")

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("开始测试正确的数据集")
    print("=" * 80)
    
    # 1. 测试数据加载
    dataset = test_data_loading()
    
    # 2. 测试vesselness
    test_vesselness(dataset)
    
    # 3. 测试奖励计算
    test_reward_calculation(dataset)
    
    print("\n" + "=" * 80)
    print("✓ 所有测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

