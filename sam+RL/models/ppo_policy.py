"""
PPO 策略网络（基于 stable-baselines3）
"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Any


class CandidateSelectionFeatureExtractor(BaseFeaturesExtractor):
    """
    候选选择任务的特征提取器
    将观察向量通过 MLP 提取特征
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        # MLP 特征提取器
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def create_ppo_model(
    env,
    config: Dict[str, Any],
    tensorboard_log: str = None,
    verbose: int = 1
) -> PPO:
    """
    创建 PPO 模型
    
    Args:
        env: RL 环境
        config: PPO 配置
        tensorboard_log: TensorBoard 日志目录
        verbose: 日志详细程度
    
    Returns:
        model: PPO 模型
    """
    ppo_config = config['ppo']
    
    # 策略参数
    policy_kwargs = dict(
        features_extractor_class=CandidateSelectionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 策略网络和价值网络的隐藏层
    )
    
    # 创建 PPO 模型
    model = PPO(
        policy=ActorCriticPolicy,
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device='auto'  # 自动选择 CUDA 或 CPU
    )
    
    return model

