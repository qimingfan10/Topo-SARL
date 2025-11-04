"""
快速测试脚本：在小数据集上快速验证系统
"""
import argparse
import os
import yaml
import numpy as np
import sys
import time

sys.path.insert(0, '/home/ubuntu/sam+RL')

from models.sam2_wrapper import SAM2CandidateGenerator
from models.ppo_policy import create_ppo_model
from rewards.reward_functions import RewardCalculator
from env.candidate_selection_env import CandidateSelectionEnv
from utils.data_loader import VesselDataset
from utils.metrics_logger import MetricsLogger

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class DetailedMonitor(Monitor):
    """带详细日志的 Monitor"""
    
    def __init__(self, env, metrics_logger=None):
        super().__init__(env)
        self.metrics_logger = metrics_logger
        self.episode_count = 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 记录到 metrics logger
        if self.metrics_logger:
            self.metrics_logger.log_step(
                step=len(self.episode_returns),
                action=action,
                reward=reward,
                info=info
            )
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 结束上一个 episode
        if self.metrics_logger and self.episode_count > 0:
            self.metrics_logger.end_episode()
        
        # 开始新 episode
        self.episode_count += 1
        if self.metrics_logger:
            self.metrics_logger.start_episode(self.episode_count, info)
        
        return obs, info


import gymnasium as gym

class TrainingDataWrapper(gym.Wrapper):
    """训练数据包装器"""
    
    def __init__(self, env, dataset, metrics_logger=None):
        super().__init__(env)
        self.dataset = dataset
        self.metrics_logger = metrics_logger
        self.sample_count = 0
    
    def reset(self, **kwargs):
        # 从数据集中随机采样
        sample = self.dataset.get_random_sample()
        self.sample_count += 1
        
        print(f"\n{'='*60}")
        print(f"Episode {self.sample_count}: {sample['name']}")
        print(f"{'='*60}")
        
        options = {
            'image': sample['image'],
            'gt_mask': sample.get('mask', None)
        }
        
        return self.env.reset(options=options)


def quick_test(args):
    """快速测试主函数"""
    
    print("\n" + "="*60)
    print("快速测试模式")
    print("="*60 + "\n")
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以加速测试
    config['training']['total_timesteps'] = args.timesteps
    config['ppo']['n_steps'] = min(512, args.timesteps // 4)  # 减小收集步数
    config['env']['max_steps'] = 5  # 减少每个 episode 的最大步数
    
    print(f"测试参数:")
    print(f"  总步数: {args.timesteps}")
    print(f"  PPO n_steps: {config['ppo']['n_steps']}")
    print(f"  每 episode 最大步数: {config['env']['max_steps']}")
    print()
    
    # 创建日志目录
    log_dir = f"./logs/quick_test_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 初始化 metrics logger
    metrics_logger = MetricsLogger(log_dir, "quick_test")
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = VesselDataset(
        image_dir=config['data']['train_image_dir'],
        mask_dir=config['data'].get('train_mask_dir'),
        image_size=tuple(config['env']['image_size'])
    )
    
    if len(dataset) == 0:
        raise ValueError(f"数据集为空: {config['data']['train_image_dir']}")
    
    print(f"✓ 加载了 {len(dataset)} 个样本\n")
    
    # 初始化 SAM2
    print("正在加载 SAM2...")
    start_time = time.time()
    sam_generator = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=config['sam2']['use_half_precision']
    )
    sam_load_time = time.time() - start_time
    metrics_logger.log_timing('sam2_load', sam_load_time)
    print(f"  加载耗时: {sam_load_time:.2f} 秒\n")
    
    # 初始化奖励计算器
    reward_calculator = RewardCalculator(config['reward'])
    
    # 创建环境
    print("正在创建环境...")
    base_env = CandidateSelectionEnv(sam_generator, reward_calculator, config)
    monitored_env = DetailedMonitor(base_env, metrics_logger)
    wrapped_env = TrainingDataWrapper(monitored_env, dataset, metrics_logger)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    print("✓ 环境创建完成\n")
    
    # 创建 PPO 模型
    print("正在创建 PPO 模型...")
    model = create_ppo_model(
        env=vec_env,
        config=config,
        tensorboard_log=log_dir,
        verbose=1
    )
    print("✓ PPO 模型创建完成\n")
    
    # 开始训练
    print("="*60)
    print("开始训练")
    print("="*60 + "\n")
    
    train_start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n训练被中断")
    
    train_duration = time.time() - train_start_time
    metrics_logger.log_timing('total_training', train_duration)
    
    # 保存模型
    model_path = f"./checkpoints/quick_test_{int(time.time())}.zip"
    model.save(model_path)
    print(f"\n✓ 模型已保存: {model_path}")
    
    # 保存指标
    metrics_logger.save()
    metrics_logger.plot_metrics()
    
    # 打印总结
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"总训练时间: {train_duration:.2f} 秒")
    print(f"日志目录: {log_dir}")
    print(f"模型路径: {model_path}")
    print(f"\n查看详细报告:")
    print(f"  cat {log_dir}/quick_test/summary_report.txt")
    print(f"  tensorboard --logdir {log_dir}")
    
    vec_env.close()
    
    return model_path, log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='快速测试')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/ubuntu/sam+RL/config/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=2048,
        help='训练步数（默认 2048，约 5-10 分钟）'
    )
    
    args = parser.parse_args()
    
    quick_test(args)

