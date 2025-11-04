"""
阶段B训练脚本：Prompt Decision Maker
"""
import argparse
import os
import yaml
from pathlib import Path
import sys
import gymnasium as gym
import time

sys.path.insert(0, '/home/ubuntu/sam+RL')

from models.sam2_wrapper import SAM2CandidateGenerator
from models.ppo_policy import create_ppo_model
from rewards.reward_functions import RewardCalculator
from env.prompt_decision_env import PromptDecisionEnv  # 新环境！
from utils.data_loader import VesselDataset
from utils.enhanced_metrics import EnhancedMetricsTracker
from utils.training_callback import EnhancedMetricsCallback

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class TrainingDataWrapper(gym.Wrapper):
    """
    训练数据包装器：每个 episode 从数据集中随机采样一个图像
    """
    
    def __init__(self, env: PromptDecisionEnv, dataset: VesselDataset):
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
        env = PromptDecisionEnv(sam_generator, reward_calculator, config)
        env = TrainingDataWrapper(env, dataset)
        env = Monitor(env)
        return env
    return _init


def train(args):
    """训练主函数"""
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("阶段 B：Prompt Decision Maker 训练")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"训练步数: {config['training']['total_timesteps']}")
    print(f"数据集: {config['data']['train_image_dir']}")
    print(f"网格大小: {config['env']['grid_size']}×{config['env']['grid_size']}")
    print(f"最大步数: {config['env']['max_steps']}")
    print("=" * 80)
    print()
    
    # 创建输出目录
    log_dir = Path(config['training']['log_dir'])
    save_dir = Path(config['training']['save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建增强的指标追踪器
    metrics_tracker = EnhancedMetricsTracker(
        window_size=100,
        save_dir=str(log_dir)
    )
    print("✓ 指标追踪器已创建\n")
    
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
    print("正在初始化奖励计算器...")
    reward_calculator = RewardCalculator(config['reward'])
    print(f"✓ 奖励配置:")
    print(f"  - 增量IoU权重: {config['reward']['delta_iou_weight']}")
    print(f"  - 最终IoU权重: {config['reward']['final_iou_weight']}")
    print(f"  - 动作成本: {config['reward']['action_cost']}")
    print()
    
    # 创建环境
    print("正在创建 RL 环境（阶段B）...")
    env = DummyVecEnv([make_env(config, sam_generator, reward_calculator, dataset)])
    print("✓ 环境创建完成\n")
    
    # 创建 PPO 模型
    print("正在创建 PPO 模型...")
    model = create_ppo_model(
        env=env,
        config=config,
        tensorboard_log=str(log_dir),
        verbose=config['logging']['verbose']
    )
    print("✓ PPO 模型创建完成\n")
    
    # 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(save_dir),
        name_prefix='ppo_stage_b'
    )
    
    metrics_callback = EnhancedMetricsCallback(
        metrics_tracker=metrics_tracker,
        print_freq=10,  # 每10个episodes打印一次
        save_freq=50,   # 每50个episodes保存一次
        verbose=1
    )
    
    callback_list = CallbackList([checkpoint_callback, metrics_callback])
    
    # 开始训练
    print("=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"总步数: {config['training']['total_timesteps']}")
    print(f"日志目录: {log_dir}")
    print(f"保存目录: {save_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callback_list,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    # 保存最终模型
    final_model_path = save_dir / 'ppo_stage_b_final.zip'
    model.save(str(final_model_path))
    
    # 打印最终统计
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"训练时间: {training_time/60:.2f} 分钟")
    print(f"最终模型: {final_model_path}")
    print()
    
    # 打印详细统计
    metrics_tracker.print_summary(last_n=100)
    
    # 保存指标
    metrics_tracker.save('final_metrics.json')
    
    env.close()
    
    return metrics_tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='阶段B训练脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/ubuntu/sam+RL/config/stage_b.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    metrics_tracker = train(args)

