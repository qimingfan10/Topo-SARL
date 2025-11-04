"""
增强的训练指标追踪系统
记录详细的训练曲线、IoU趋势、动作分布等
"""
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any
import json
from pathlib import Path


class EnhancedMetricsTracker:
    """
    增强的指标追踪器
    """
    
    def __init__(self, window_size: int = 100, save_dir: str = "./logs"):
        """
        Args:
            window_size: 移动平均窗口大小
            save_dir: 保存目录
        """
        self.window_size = window_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 基本统计
        self.total_steps = 0
        self.total_episodes = 0
        
        # 奖励追踪
        self.episode_rewards = []
        self.step_rewards = []
        self.reward_components = defaultdict(list)  # iou, cldice, etc.
        
        # IoU追踪
        self.episode_ious = []
        self.step_ious = []
        self.best_iou = 0.0
        self.best_iou_episode = 0
        
        # 动作统计
        self.action_counts = defaultdict(int)
        self.action_history = []
        
        # Episode统计
        self.episode_lengths = []
        self.episode_final_ious = []
        self.episode_final_areas = []  # 最终掩膜面积
        
        # 移动平均
        self.reward_window = deque(maxlen=window_size)
        self.iou_window = deque(maxlen=window_size)
        
        # 当前episode缓存
        self.current_episode = {
            'rewards': [],
            'ious': [],
            'actions': [],
            'reward_components': defaultdict(list)
        }
        
        # SAM2统计
        self.sam2_inference_times = []
        self.candidates_generated = []
        
        # 训练阶段统计
        self.phase_stats = {
            'early': {'episodes': 0, 'avg_reward': 0, 'avg_iou': 0},  # 前1000步
            'mid': {'episodes': 0, 'avg_reward': 0, 'avg_iou': 0},    # 1000-5000步
            'late': {'episodes': 0, 'avg_reward': 0, 'avg_iou': 0}    # 5000+步
        }
    
    def log_step(self, 
                 reward: float, 
                 action: int, 
                 iou: float = None,
                 reward_components: Dict[str, float] = None,
                 **kwargs):
        """
        记录单步信息
        
        Args:
            reward: 步奖励
            action: 动作ID
            iou: 当前IoU（如果有GT）
            reward_components: 奖励各组成部分
            **kwargs: 其他信息
        """
        self.total_steps += 1
        self.step_rewards.append(reward)
        
        # 记录动作
        self.action_counts[action] += 1
        self.action_history.append(action)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        
        # 记录IoU
        if iou is not None:
            self.step_ious.append(iou)
            self.current_episode['ious'].append(iou)
            self.iou_window.append(iou)
        
        # 记录奖励组件
        if reward_components:
            for key, value in reward_components.items():
                self.reward_components[key].append(value)
                self.current_episode['reward_components'][key].append(value)
        
        # 记录其他信息
        if 'sam2_time' in kwargs:
            self.sam2_inference_times.append(kwargs['sam2_time'])
        if 'num_candidates' in kwargs:
            self.candidates_generated.append(kwargs['num_candidates'])
    
    def log_episode_end(self, final_iou: float = None, final_area: int = None):
        """
        记录episode结束
        
        Args:
            final_iou: 最终IoU
            final_area: 最终掩膜面积
        """
        self.total_episodes += 1
        
        # 计算episode总奖励
        episode_reward = sum(self.current_episode['rewards'])
        self.episode_rewards.append(episode_reward)
        self.reward_window.append(episode_reward)
        
        # 记录episode长度
        episode_length = len(self.current_episode['rewards'])
        self.episode_lengths.append(episode_length)
        
        # 记录最终IoU
        if final_iou is not None:
            self.episode_final_ious.append(final_iou)
            self.episode_ious.append(final_iou)
            
            # 更新最佳IoU
            if final_iou > self.best_iou:
                self.best_iou = final_iou
                self.best_iou_episode = self.total_episodes
        
        # 记录最终面积
        if final_area is not None:
            self.episode_final_areas.append(final_area)
        
        # 更新训练阶段统计
        self._update_phase_stats(episode_reward, final_iou)
        
        # 清空当前episode缓存
        self.current_episode = {
            'rewards': [],
            'ious': [],
            'actions': [],
            'reward_components': defaultdict(list)
        }
    
    def _update_phase_stats(self, reward: float, iou: float = None):
        """更新训练阶段统计"""
        if self.total_steps < 1000:
            phase = 'early'
        elif self.total_steps < 5000:
            phase = 'mid'
        else:
            phase = 'late'
        
        stats = self.phase_stats[phase]
        n = stats['episodes']
        stats['episodes'] += 1
        stats['avg_reward'] = (stats['avg_reward'] * n + reward) / (n + 1)
        if iou is not None:
            stats['avg_iou'] = (stats['avg_iou'] * n + iou) / (n + 1)
    
    def get_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """
        获取统计摘要
        
        Args:
            last_n: 最近N个episodes
            
        Returns:
            summary: 统计摘要字典
        """
        summary = {}
        
        # 基本统计
        summary['total_steps'] = self.total_steps
        summary['total_episodes'] = self.total_episodes
        
        # 奖励统计
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-last_n:]
            summary['avg_episode_reward'] = np.mean(recent_rewards)
            summary['std_episode_reward'] = np.std(recent_rewards)
            summary['min_episode_reward'] = np.min(recent_rewards)
            summary['max_episode_reward'] = np.max(recent_rewards)
            summary['moving_avg_reward'] = np.mean(list(self.reward_window)) if self.reward_window else 0
        
        # IoU统计
        if self.episode_final_ious:
            recent_ious = self.episode_final_ious[-last_n:]
            summary['avg_final_iou'] = np.mean(recent_ious)
            summary['std_final_iou'] = np.std(recent_ious)
            summary['min_final_iou'] = np.min(recent_ious)
            summary['max_final_iou'] = np.max(recent_ious)
            summary['best_iou'] = self.best_iou
            summary['best_iou_episode'] = self.best_iou_episode
            summary['moving_avg_iou'] = np.mean(list(self.iou_window)) if self.iou_window else 0
        
        # 动作统计
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            summary['action_distribution'] = {
                k: v / total_actions for k, v in self.action_counts.items()
            }
            summary['action_counts'] = dict(self.action_counts)
        
        # Episode长度统计
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-last_n:]
            summary['avg_episode_length'] = np.mean(recent_lengths)
            summary['std_episode_length'] = np.std(recent_lengths)
        
        # 掩膜面积统计
        if self.episode_final_areas:
            recent_areas = self.episode_final_areas[-last_n:]
            summary['avg_final_area'] = np.mean(recent_areas)
            summary['std_final_area'] = np.std(recent_areas)
        
        # SAM2统计
        if self.sam2_inference_times:
            recent_times = [t for t in self.sam2_inference_times[-last_n:] if t is not None]
            if recent_times:
                summary['avg_sam2_time'] = np.mean(recent_times)
        
        if self.candidates_generated:
            recent_candidates = [c for c in self.candidates_generated[-last_n:] if c is not None]
            if recent_candidates:
                summary['avg_candidates'] = np.mean(recent_candidates)
        
        # 训练阶段统计
        summary['phase_stats'] = self.phase_stats
        
        # 奖励组件统计
        if self.reward_components:
            summary['reward_components'] = {}
            for key, values in self.reward_components.items():
                recent_values = values[-last_n*10:]  # 每个episode可能有多步
                if recent_values:
                    summary['reward_components'][key] = {
                        'mean': float(np.mean(recent_values)),
                        'std': float(np.std(recent_values))
                    }
        
        return summary
    
    def print_summary(self, last_n: int = 100):
        """打印统计摘要"""
        summary = self.get_summary(last_n)
        
        print("\n" + "=" * 80)
        print(f"训练统计摘要 (最近{last_n}个episodes)")
        print("=" * 80)
        
        # 基本信息
        print(f"\n📊 基本统计:")
        print(f"  - 总步数: {summary['total_steps']}")
        print(f"  - 总Episodes: {summary['total_episodes']}")
        
        # 奖励
        if 'avg_episode_reward' in summary:
            print(f"\n💰 奖励:")
            print(f"  - 平均奖励: {summary['avg_episode_reward']:.4f} ± {summary['std_episode_reward']:.4f}")
            print(f"  - 奖励范围: [{summary['min_episode_reward']:.4f}, {summary['max_episode_reward']:.4f}]")
            print(f"  - 移动平均: {summary['moving_avg_reward']:.4f}")
        
        # IoU
        if 'avg_final_iou' in summary:
            print(f"\n🎯 IoU:")
            print(f"  - 平均IoU: {summary['avg_final_iou']:.4f} ± {summary['std_final_iou']:.4f}")
            print(f"  - IoU范围: [{summary['min_final_iou']:.4f}, {summary['max_final_iou']:.4f}]")
            print(f"  - 最佳IoU: {summary['best_iou']:.4f} (Episode {summary['best_iou_episode']})")
            print(f"  - 移动平均: {summary['moving_avg_iou']:.4f}")
        
        # 动作
        if 'action_distribution' in summary:
            print(f"\n🎬 动作分布:")
            action_names = {0: 'select', 1: 'sample', 2: 'merge', 3: 'terminate'}
            for action_id, ratio in summary['action_distribution'].items():
                action_name = action_names.get(action_id, f'action_{action_id}')
                count = summary['action_counts'][action_id]
                print(f"  - {action_name}: {ratio*100:.1f}% ({count}次)")
        
        # Episode长度
        if 'avg_episode_length' in summary:
            print(f"\n📏 Episode长度:")
            print(f"  - 平均: {summary['avg_episode_length']:.2f} ± {summary['std_episode_length']:.2f}")
        
        # 掩膜面积
        if 'avg_final_area' in summary:
            print(f"\n📐 最终掩膜面积:")
            print(f"  - 平均: {summary['avg_final_area']:.0f} ± {summary['std_final_area']:.0f} 像素")
            print(f"  - 占比: {summary['avg_final_area']/(512*512)*100:.2f}%")
        
        # 训练阶段
        print(f"\n📈 训练阶段:")
        for phase, stats in summary['phase_stats'].items():
            if stats['episodes'] > 0:
                print(f"  - {phase}: {stats['episodes']}个episodes, "
                      f"平均奖励={stats['avg_reward']:.4f}, "
                      f"平均IoU={stats['avg_iou']:.4f}")
        
        # 奖励组件
        if 'reward_components' in summary and summary['reward_components']:
            print(f"\n🔍 奖励组件:")
            for key, stats in summary['reward_components'].items():
                print(f"  - {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print("=" * 80)
    
    def save(self, filename: str = "metrics.json"):
        """保存指标到文件"""
        filepath = self.save_dir / filename
        
        # 转换所有numpy类型为Python原生类型
        def convert_to_native(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {str(k): convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        data = {
            'total_steps': int(self.total_steps),
            'total_episodes': int(self.total_episodes),
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_ious': [float(x) for x in self.episode_ious],
            'episode_lengths': [int(x) for x in self.episode_lengths],
            'episode_final_ious': [float(x) for x in self.episode_final_ious],
            'episode_final_areas': [int(x) for x in self.episode_final_areas],
            'action_counts': {str(k): int(v) for k, v in self.action_counts.items()},
            'best_iou': float(self.best_iou),
            'best_iou_episode': int(self.best_iou_episode),
            'phase_stats': convert_to_native(self.phase_stats),
            'reward_components': {k: [float(x) for x in v] for k, v in self.reward_components.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ 指标已保存到: {filepath}")
    
    def load(self, filename: str = "metrics.json"):
        """从文件加载指标"""
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ 文件不存在: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.total_steps = data['total_steps']
        self.total_episodes = data['total_episodes']
        self.episode_rewards = data['episode_rewards']
        self.episode_ious = data['episode_ious']
        self.episode_lengths = data['episode_lengths']
        self.episode_final_ious = data['episode_final_ious']
        self.episode_final_areas = data['episode_final_areas']
        self.action_counts = defaultdict(int, data['action_counts'])
        self.best_iou = data['best_iou']
        self.best_iou_episode = data['best_iou_episode']
        self.phase_stats = data['phase_stats']
        self.reward_components = defaultdict(list, data['reward_components'])
        
        print(f"✓ 指标已从文件加载: {filepath}")

