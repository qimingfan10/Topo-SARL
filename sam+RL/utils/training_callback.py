"""
自定义训练回调，集成增强的指标追踪
"""
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any
import numpy as np
from .enhanced_metrics import EnhancedMetricsTracker


class EnhancedMetricsCallback(BaseCallback):
    """
    增强的指标追踪回调
    """
    
    def __init__(self, metrics_tracker: EnhancedMetricsTracker, 
                 print_freq: int = 10, 
                 save_freq: int = 100,
                 verbose: int = 0):
        """
        Args:
            metrics_tracker: 指标追踪器
            print_freq: 打印频率（每N个episodes）
            save_freq: 保存频率（每N个episodes）
            verbose: 详细级别
        """
        super().__init__(verbose)
        self.metrics_tracker = metrics_tracker
        self.print_freq = print_freq
        self.save_freq = save_freq
        
        # Episode追踪
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.episodes_since_print = 0
        self.episodes_since_save = 0
        
        # 从环境info中获取的信息
        self.last_info = {}
    
    def _on_step(self) -> bool:
        """
        每步后调用
        """
        # 获取当前奖励
        if len(self.locals['rewards']) > 0:
            reward = self.locals['rewards'][0]
            self.current_episode_reward += reward
            self.current_episode_steps += 1
            
            # 获取info
            if len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                self.last_info = info
                
                # 记录步信息
                self.metrics_tracker.log_step(
                    reward=reward,
                    action=self.locals['actions'][0] if 'actions' in self.locals else -1,
                    iou=info.get('current_iou', None),
                    reward_components=info.get('reward_components', None),
                    sam2_time=info.get('sam2_time', None),
                    num_candidates=info.get('num_candidates', None)
                )
        
        # 检查episode是否结束
        if len(self.locals['dones']) > 0 and self.locals['dones'][0]:
            # Episode结束
            final_iou = self.last_info.get('final_iou', None)
            final_area = self.last_info.get('final_area', None)
            
            self.metrics_tracker.log_episode_end(
                final_iou=final_iou,
                final_area=final_area
            )
            
            self.episodes_since_print += 1
            self.episodes_since_save += 1
            
            # 打印统计
            if self.episodes_since_print >= self.print_freq:
                self.metrics_tracker.print_summary(last_n=self.print_freq)
                self.episodes_since_print = 0
            
            # 保存指标
            if self.episodes_since_save >= self.save_freq:
                self.metrics_tracker.save()
                self.episodes_since_save = 0
            
            # 重置episode计数
            self.current_episode_reward = 0
            self.current_episode_steps = 0
            self.last_info = {}
        
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时调用"""
        print("\n" + "=" * 80)
        print("训练结束 - 最终统计")
        print("=" * 80)
        self.metrics_tracker.print_summary(last_n=100)
        self.metrics_tracker.save()

