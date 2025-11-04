"""
测试阶段B环境：PromptDecisionEnv
验证环境基础功能是否正常
"""
import sys
import yaml
import numpy as np

sys.path.insert(0, '/home/ubuntu/sam+RL')

from env.prompt_decision_env import PromptDecisionEnv
from models.sam2_wrapper import SAM2CandidateGenerator
from rewards.reward_functions import RewardCalculator
from utils.data_loader import VesselDataset


def test_basic_functionality():
    """测试基础功能"""
    print("=" * 80)
    print("测试阶段B环境基础功能")
    print("=" * 80)
    
    # 加载配置
    print("\n1. 加载配置...")
    config = {
        'sam2': {
            'checkpoint': '/home/ubuntu/sam2.1_hiera_large.pt',
            'model_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            'device': 'cuda',
            'use_half_precision': True
        },
        'env': {
            'max_steps': 20,
            'grid_size': 32,
            'image_size': [512, 512],
            'max_points': 20
        },
        'reward': {
            'use_gt': True,
            'delta_iou_weight': 10.0,
            'final_iou_weight': 5.0,
            'action_cost': -0.01,
            'iou_decrease_penalty': -0.5
        }
    }
    print("✓ 配置加载完成")
    
    # 初始化组件
    print("\n2. 初始化SAM2...")
    sam_generator = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=config['sam2']['use_half_precision']
    )
    
    print("\n3. 初始化奖励计算器...")
    reward_calculator = RewardCalculator(config['reward'])
    print("✓ 奖励计算器初始化完成")
    
    # 创建环境
    print("\n4. 创建环境...")
    env = PromptDecisionEnv(sam_generator, reward_calculator, config)
    
    # 加载测试数据
    print("\n5. 加载测试数据...")
    dataset = VesselDataset(
        image_dir="/home/ubuntu/Segment_DATA/orgin_pic",
        mask_dir="/home/ubuntu/Segment_DATA/lab_pic",
        image_size=(512, 512)
    )
    sample = dataset[0]
    print(f"✓ 测试样本: {sample['name']}")
    print(f"  - 图像形状: {sample['image'].shape}")
    print(f"  - 掩膜形状: {sample['mask'].shape}")
    print(f"  - 血管面积: {sample['mask'].sum()} 像素 ({sample['mask'].sum()/(512*512)*100:.2f}%)")
    
    # 测试reset
    print("\n6. 测试 reset()...")
    obs, info = env.reset(options={'image': sample['image'], 'gt_mask': sample['mask']})
    print(f"✓ Reset成功")
    print(f"  - 观察空间维度: {obs.shape}")
    print(f"  - 初始IoU: {info['current_iou']:.4f}")
    print(f"  - 步数: {info['step_count']}")
    
    # 测试动作执行
    print("\n7. 测试随机动作执行...")
    total_reward = 0.0
    for step in range(5):
        # 随机采样动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\n  步骤 {step + 1}:")
        print(f"    - 动作: {action} -> {info['action_name']}")
        if info['action_name'] != 'terminate':
            print(f"    - 位置: 网格{info['grid_pos']}") 
        print(f"    - 当前IoU: {info.get('current_iou', 0):.4f}")
        print(f"    - 奖励: {reward:.4f}")
        print(f"    - 提示点数: {info.get('num_points', 0)}")
        
        if terminated or truncated:
            print(f"    - Episode结束")
            break
    
    print(f"\n  总奖励: {total_reward:.4f}")
    
    # 测试动作解码
    print("\n8. 测试动作编解码...")
    test_actions = [
        0,      # add_positive at (0,0)
        1023,   # add_positive at (31,31)
        1024,   # add_negative at (0,0)
        2047,   # add_negative at (31,31)
        2048,   # terminate at (0,0)
        3071    # terminate at (31,31)
    ]
    
    for action in test_actions:
        action_type, grid_x, grid_y = env._decode_action(action)
        action_names = ['positive', 'negative', 'terminate']
        print(f"  动作 {action:4d} -> {action_names[action_type]:10s} at grid ({grid_x:2d}, {grid_y:2d})")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！环境工作正常")
    print("=" * 80)
    
    return env


def test_full_episode():
    """测试完整episode"""
    print("\n" + "=" * 80)
    print("测试完整Episode")
    print("=" * 80)
    
    # 创建环境（复用前面的逻辑）
    config = {
        'sam2': {
            'checkpoint': '/home/ubuntu/sam2.1_hiera_large.pt',
            'model_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            'device': 'cuda',
            'use_half_precision': True
        },
        'env': {
            'max_steps': 10,  # 减少步数加快测试
            'grid_size': 32,
            'image_size': [512, 512],
            'max_points': 20
        },
        'reward': {
            'use_gt': True,
            'delta_iou_weight': 10.0,
            'final_iou_weight': 5.0,
            'action_cost': -0.01,
            'iou_decrease_penalty': -0.5
        }
    }
    
    sam_generator = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=config['sam2']['use_half_precision']
    )
    
    reward_calculator = RewardCalculator(config['reward'])
    env = PromptDecisionEnv(sam_generator, reward_calculator, config)
    
    # 加载数据
    dataset = VesselDataset(
        image_dir="/home/ubuntu/Segment_DATA/orgin_pic",
        mask_dir="/home/ubuntu/Segment_DATA/lab_pic",
        image_size=(512, 512)
    )
    
    # 运行完整episode
    print("\n运行完整Episode...")
    sample = dataset[0]
    obs, info = env.reset(options={'image': sample['image'], 'gt_mask': sample['mask']})
    
    episode_reward = 0.0
    best_iou = 0.0
    
    for step in range(config['env']['max_steps']):
        # 随机策略（后续会用训练好的策略）
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        current_iou = info.get('current_iou', 0.0)
        if current_iou > best_iou:
            best_iou = current_iou
        
        print(f"  步骤 {step + 1}: {info['action_name']:10s} | IoU={current_iou:.4f} | 奖励={reward:+.4f}")
        
        if terminated or truncated:
            print(f"\n  Episode结束:")
            print(f"    - 最终IoU: {info.get('final_iou', 0):.4f}")
            print(f"    - 最佳IoU: {best_iou:.4f}")
            print(f"    - 总奖励: {episode_reward:.4f}")
            print(f"    - 总步数: {step + 1}")
            print(f"    - 提示点数: {info.get('num_points', 0)}")
            break
    
    print("\n✅ 完整Episode测试通过！")
    
    return env


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("阶段B环境测试")
    print("=" * 80)
    
    # 测试1：基础功能
    env = test_basic_functionality()
    
    # 测试2：完整Episode
    env = test_full_episode()
    
    print("\n" + "=" * 80)
    print("🎉 所有测试完成！阶段B环境准备就绪")
    print("=" * 80)
    print("\n下一步：")
    print("1. 优化观察空间（添加CNN特征提取）")
    print("2. 创建训练脚本")
    print("3. 运行初步训练测试")
    print()

