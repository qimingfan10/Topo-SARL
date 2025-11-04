"""
训练脚本：阶段 A - 候选选择器
"""
import argparse
import os
import yaml
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym

sys.path.insert(0, '/home/ubuntu/sam+RL')

from models.sam2_wrapper import SAM2CandidateGenerator
from models.ppo_policy import create_ppo_model
from rewards.reward_functions import RewardCalculator
from env.candidate_selection_env import CandidateSelectionEnv
from utils.data_loader import VesselDataset

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class TrainingDataWrapper(gym.Wrapper):
    """
    训练数据包装器：每个 episode 从数据集中随机采样一个图像
    """
    
    def __init__(self, env: CandidateSelectionEnv, dataset: VesselDataset):
        super().__init__(env)
        self.dataset = dataset
    
    def reset(self, **kwargs):
        # 从数据集中随机采样
        sample = self.dataset.get_random_sample()
        
        # 传递给环境
        options = {
            'image': sample['image'],
            'gt_mask': sample.get('mask', None)
        }
        
        return self.env.reset(options=options)
    
    def step(self, action):
        return self.env.step(action)


def make_env(config, sam_generator, reward_calculator, dataset):
    """创建环境实例"""
    def _init():
        env = CandidateSelectionEnv(sam_generator, reward_calculator, config)
        env = TrainingDataWrapper(env, dataset)
        env = Monitor(env)
        return env
    return _init


def train(args):
    """训练主函数"""
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("========================================")
    print("阶段 A：候选选择器训练")
    print("========================================\n")
    
    # 创建输出目录
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = VesselDataset(
        image_dir=config['data']['train_image_dir'],
        mask_dir=config['data'].get('train_mask_dir'),
        image_size=tuple(config['env']['image_size'])
    )
    
    if len(dataset) == 0:
        raise ValueError(f"数据集为空，请检查路径: {config['data']['train_image_dir']}")
    
    print()
    
    # 初始化 SAM2
    print("正在加载 SAM2 模型...")
    sam_generator = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=config['sam2']['use_half_precision']
    )
    print()
    
    # 初始化奖励计算器
    reward_calculator = RewardCalculator(config['reward'])
    
    # 创建环境
    print("正在创建 RL 环境...")
    env = DummyVecEnv([make_env(config, sam_generator, reward_calculator, dataset)])
    print("✓ 环境创建完成\n")
    
    # 创建 PPO 模型
    print("正在创建 PPO 模型...")
    model = create_ppo_model(
        env=env,
        config=config,
        tensorboard_log=config['training']['log_dir'],
        verbose=config['logging']['verbose']
    )
    print("✓ PPO 模型创建完成\n")
    
    # 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=config['training']['save_dir'],
        name_prefix='ppo_candidate_selector'
    )
    
    # 开始训练
    print("========================================")
    print("开始训练")
    print("========================================\n")
    print(f"总步数: {config['training']['total_timesteps']}")
    print(f"日志目录: {config['training']['log_dir']}")
    print(f"保存目录: {config['training']['save_dir']}")
    print(f"TensorBoard: tensorboard --logdir {config['training']['log_dir']}\n")
    
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(config['training']['save_dir'], 'ppo_candidate_selector_final.zip')
    model.save(final_model_path)
    
    print("\n========================================")
    print("训练完成！")
    print("========================================")
    print(f"最终模型已保存: {final_model_path}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练候选选择器（阶段 A）')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/ubuntu/sam+RL/config/default.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    train(args)

