"""
模块测试脚本：验证各个组件是否正常工作
"""
import sys
sys.path.insert(0, '/home/ubuntu/sam+RL')
sys.path.insert(0, '/home/ubuntu/sam2')

import numpy as np
import cv2
import yaml


def test_sam2_wrapper():
    """测试 SAM2 包装器"""
    print("\n========================================")
    print("测试 SAM2 包装器")
    print("========================================")
    
    from models.sam2_wrapper import SAM2CandidateGenerator
    
    # 加载配置
    with open('/home/ubuntu/sam+RL/config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化
    sam_gen = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=config['sam2']['use_half_precision']
    )
    
    # 创建测试图像（512x512，随机图案）
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 设置图像
    sam_gen.set_image(test_image)
    print("✓ 图像设置成功")
    
    # 采样随机点
    points, labels = sam_gen.sample_random_points(num_points=5)
    print(f"✓ 随机采样 {len(points)} 个点")
    
    # 生成候选
    candidates = sam_gen.generate_candidates(
        points=points,
        point_labels=labels,
        mask_thresholds=[0.0, 0.5, 1.0],
        multimask_output=True
    )
    print(f"✓ 生成 {len(candidates)} 个候选掩膜")
    
    # 验证候选格式
    for i, cand in enumerate(candidates[:3]):
        assert 'mask' in cand
        assert 'score' in cand
        assert cand['mask'].shape == (512, 512)
        print(f"  候选 {i}: 面积={cand['mask'].sum()}, 分数={cand['score']:.3f}")
    
    print("✓ SAM2 包装器测试通过")


def test_reward_functions():
    """测试奖励函数"""
    print("\n========================================")
    print("测试奖励函数")
    print("========================================")
    
    from rewards.reward_functions import (
        compute_iou, compute_dice, compute_cldice, 
        compute_topology_penalty, RewardCalculator
    )
    
    # 创建测试掩膜
    pred = np.zeros((512, 512), dtype=bool)
    pred[100:300, 100:300] = True
    
    gt = np.zeros((512, 512), dtype=bool)
    gt[150:350, 150:350] = True
    
    # 测试各项指标
    iou = compute_iou(pred, gt)
    dice = compute_dice(pred, gt)
    cldice = compute_cldice(pred, gt)
    topo_penalty = compute_topology_penalty(pred)
    
    print(f"✓ IoU: {iou:.4f}")
    print(f"✓ Dice: {dice:.4f}")
    print(f"✓ clDice: {cldice:.4f}")
    print(f"✓ 拓扑惩罚: {topo_penalty:.4f}")
    
    # 测试奖励计算器
    config = {
        'use_gt': True,
        'iou_weight': 1.0,
        'cldice_weight': 0.5,
        'topology_weight': 0.3,
        'action_cost': -0.01
    }
    calculator = RewardCalculator(config)
    
    reward_dict = calculator.compute_reward(pred, gt)
    print(f"✓ 总奖励: {reward_dict['total']:.4f}")
    print("✓ 奖励函数测试通过")


def test_environment():
    """测试 RL 环境"""
    print("\n========================================")
    print("测试 RL 环境")
    print("========================================")
    
    from models.sam2_wrapper import SAM2CandidateGenerator
    from rewards.reward_functions import RewardCalculator
    from env.candidate_selection_env import CandidateSelectionEnv
    
    # 加载配置
    with open('/home/ubuntu/sam+RL/config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化组件
    sam_gen = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=False  # 测试时不使用半精度
    )
    
    reward_calc = RewardCalculator(config['reward'])
    
    # 创建环境
    env = CandidateSelectionEnv(sam_gen, reward_calc, config)
    print("✓ 环境创建成功")
    
    # 测试 reset
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_mask = np.zeros((512, 512), dtype=bool)
    test_mask[100:300, 100:300] = True
    
    obs, info = env.reset(options={'image': test_image, 'gt_mask': test_mask})
    print(f"✓ 环境重置成功，观察维度: {obs.shape}")
    print(f"  初始候选数: {info['candidates']}")
    
    # 测试 step
    action = 0  # 选择第一个候选
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ 执行动作成功")
    print(f"  奖励: {reward:.4f}")
    print(f"  动作: {info.get('action', 'unknown')}")
    
    # 测试 render
    vis = env.render()
    print(f"✓ 渲染成功，可视化尺寸: {vis.shape}")
    
    env.close()
    print("✓ RL 环境测试通过")


def test_data_loader():
    """测试数据加载器"""
    print("\n========================================")
    print("测试数据加载器")
    print("========================================")
    
    from utils.data_loader import VesselDataset
    
    # 加载配置
    with open('/home/ubuntu/sam+RL/config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建数据集
    dataset = VesselDataset(
        image_dir=config['data']['train_image_dir'],
        mask_dir=config['data'].get('train_mask_dir'),
        image_size=tuple(config['env']['image_size'])
    )
    
    if len(dataset) > 0:
        print(f"✓ 数据集加载成功，共 {len(dataset)} 个样本")
        
        # 测试加载第一个样本
        sample = dataset[0]
        print(f"✓ 样本加载成功")
        print(f"  图像形状: {sample['image'].shape}")
        print(f"  图像名称: {sample['name']}")
        if 'mask' in sample:
            print(f"  掩膜形状: {sample['mask'].shape}")
        
        print("✓ 数据加载器测试通过")
    else:
        print("⚠️  数据集为空，请检查数据路径")


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("SAM2 + RL 模块测试")
    print("=" * 50)
    
    try:
        test_sam2_wrapper()
        test_reward_functions()
        test_environment()
        test_data_loader()
        
        print("\n" + "=" * 50)
        print("✓ 所有测试通过！")
        print("=" * 50)
        print("\n系统已准备就绪，可以开始训练。")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

