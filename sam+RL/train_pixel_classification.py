#!/usr/bin/env python3
"""
像素级分类训练脚本 - 任务重定向方案
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/sam+RL')

from env.pixel_classification_env import PixelClassificationEnv
from utils.data_loader import VesselDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch
import gymnasium as gym


class PixelClassificationWrapper(gym.Env):
    """环境wrapper，适配SB3"""
    def __init__(self, config, dataset):
        super().__init__()
        self.env = PixelClassificationEnv(config)
        self.dataset = dataset
        self.current_idx = 0
        
        # 必须设置这些属性
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self, seed=None, options=None):
        # 随机选择一个图像
        self.current_idx = np.random.randint(0, len(self.dataset))
        data = self.dataset[self.current_idx]
        # data是字典 {'image': ..., 'mask': ..., 'name': ...}
        image = data['image']
        mask = data['mask']
        obs = self.env.reset(image, mask)
        return obs, {}
    
    def step(self, action):
        return self.env.step(action)


class MetricsCallback(BaseCallback):
    """指标追踪callback"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_ious = []
        self.episode_accuracies = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 检查episode是否结束
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            env = self.training_env.envs[0].env
            
            if hasattr(env, 'episode_stats'):
                stats = env.episode_stats
                if stats['iou_history']:
                    iou = stats['iou_history'][-1]
                    self.episode_ious.append(iou)
                    
                    correct = stats['correct_classifications']
                    total = correct + stats['wrong_classifications']
                    accuracy = correct / total if total > 0 else 0
                    self.episode_accuracies.append(accuracy)
                    
                    self.episode_lengths.append(env.step_count)
                    
                    # 每10个episodes打印一次
                    if len(self.episode_ious) % 10 == 0:
                        recent_iou = np.mean(self.episode_ious[-10:]) * 100
                        recent_acc = np.mean(self.episode_accuracies[-10:]) * 100
                        recent_len = np.mean(self.episode_lengths[-10:])
                        print(f"\n[PROGRESS] Episodes:{len(self.episode_ious)}, "
                              f"Avg IoU:{recent_iou:.2f}%, Acc:{recent_acc:.1f}%, Len:{recent_len:.1f}")
        
        return True


def train(config_path: str):
    """训练主函数"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*80}")
    print(f"🚀 像素级分类训练 - 任务重定向方案")
    print(f"{'='*80}\n")
    
    print(f"📋 配置:")
    print(f"  Grid大小: {config['env']['grid_size']}x{config['env']['grid_size']}")
    print(f"  最大步数: {config['env']['max_steps']}")
    print(f"  训练步数: {config['training']['total_timesteps']}")
    print(f"  调试模式: {config.get('debug_mode', False)}")
    
    # 加载数据集
    print(f"\n📂 加载数据集...")
    dataset = VesselDataset(
        image_dir=config['data']['train_image_dir'],
        mask_dir=config['data']['train_mask_dir']
    )
    print(f"  ✓ 数据集大小: {len(dataset)}个图像")
    
    # 创建环境
    print(f"\n🏗️  创建环境...")
    def make_env():
        return PixelClassificationWrapper(config, dataset)
    
    env = DummyVecEnv([make_env])
    print(f"  ✓ 环境创建成功")
    
    # 创建模型
    print(f"\n🤖 创建PPO模型...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['ppo']['learning_rate'],
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_range=config['ppo']['clip_range'],
        ent_coef=config['ppo']['ent_coef'],
        vf_coef=config['ppo']['vf_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        verbose=config['logging']['verbose'],
        tensorboard_log=config['training']['log_dir'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"  ✓ 模型创建成功 (device: {model.device})")
    
    # 创建callbacks
    Path(config['training']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=config['training']['save_dir'],
        name_prefix='pixel_classification'
    )
    
    metrics_callback = MetricsCallback(verbose=1)
    
    # 开始训练
    print(f"\n{'='*80}")
    print(f"🎯 开始训练...")
    print(f"{'='*80}\n")
    
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = Path(config['training']['save_dir']) / 'final_model.zip'
    model.save(final_model_path)
    print(f"\n✓ 模型已保存: {final_model_path}")
    
    # 打印最终统计
    print(f"\n{'='*80}")
    print(f"🏁 训练完成")
    print(f"{'='*80}\n")
    
    if metrics_callback.episode_ious:
        avg_iou = np.mean(metrics_callback.episode_ious) * 100
        best_iou = np.max(metrics_callback.episode_ious) * 100
        avg_acc = np.mean(metrics_callback.episode_accuracies) * 100
        avg_len = np.mean(metrics_callback.episode_lengths)
        
        print(f"📊 总体统计:")
        print(f"  Episodes: {len(metrics_callback.episode_ious)}")
        print(f"  平均IoU: {avg_iou:.2f}%")
        print(f"  最佳IoU: {best_iou:.2f}%")
        print(f"  平均准确率: {avg_acc:.1f}%")
        print(f"  平均步数: {avg_len:.1f}")
        
        # 最近performance
        if len(metrics_callback.episode_ious) >= 20:
            recent_iou = np.mean(metrics_callback.episode_ious[-20:]) * 100
            recent_acc = np.mean(metrics_callback.episode_accuracies[-20:]) * 100
            print(f"\n  最近20个episodes:")
            print(f"    平均IoU: {recent_iou:.2f}%")
            print(f"    平均准确率: {recent_acc:.1f}%")
            
            if recent_iou >= 10.0:
                print(f"\n  🎉 达到10% IoU目标！")
            elif recent_iou >= 8.0:
                print(f"\n  ✅ 接近目标，继续训练可能达到10%")
            elif recent_iou >= 5.0:
                print(f"\n  ⚠️  有进展，但需要更多训练")
            else:
                print(f"\n  ⚠️  性能较低，可能需要调整配置")
    
    return metrics_callback


if __name__ == "__main__":
    config_path = '/home/ubuntu/sam+RL/config/pixel_classification_v1.yaml'
    
    try:
        metrics = train(config_path)
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

