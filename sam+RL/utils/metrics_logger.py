"""
详细的指标记录器：用于调试和分析
"""
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt


class MetricsLogger:
    """
    详细的指标记录器
    记录训练/推理过程中的各种指标，用于调试和分析
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 指标存储
        self.episode_metrics = []  # 每个 episode 的指标
        self.step_metrics = []     # 每个 step 的指标
        self.sam2_metrics = []     # SAM2 相关指标
        self.timing_metrics = []   # 时间统计
        
        # 当前 episode 的临时存储
        self.current_episode = {
            'steps': [],
            'start_time': None,
            'episode_id': 0
        }
        
        print(f"✓ 指标记录器已初始化: {self.log_dir}")
    
    def start_episode(self, episode_id: int, info: Dict = None):
        """开始一个新 episode"""
        self.current_episode = {
            'episode_id': episode_id,
            'steps': [],
            'start_time': time.time(),
            'info': info or {}
        }
    
    def log_step(
        self,
        step: int,
        action: int,
        reward: float,
        info: Dict[str, Any]
    ):
        """记录一个 step 的指标"""
        step_data = {
            'step': step,
            'action': action,
            'action_name': info.get('action', 'unknown'),
            'reward': reward,
            'timestamp': time.time() - self.current_episode['start_time']
        }
        
        # 提取奖励组成
        for key in ['iou', 'dice', 'cldice', 'topology_penalty', 
                    'vesselness', 'fp_penalty', 'smoothness', 'action_cost']:
            if key in info:
                step_data[key] = info[key]
        
        # 其他信息
        if 'merged_count' in info:
            step_data['merged_count'] = info['merged_count']
        if 'new_candidates' in info:
            step_data['new_candidates'] = info['new_candidates']
        
        self.current_episode['steps'].append(step_data)
        self.step_metrics.append(step_data)
    
    def end_episode(self, final_info: Dict = None):
        """结束当前 episode 并汇总"""
        episode_summary = {
            'episode_id': self.current_episode['episode_id'],
            'total_steps': len(self.current_episode['steps']),
            'total_reward': sum(s['reward'] for s in self.current_episode['steps']),
            'duration': time.time() - self.current_episode['start_time'],
            'final_info': final_info or {}
        }
        
        # 计算各项指标的平均值
        if self.current_episode['steps']:
            steps = self.current_episode['steps']
            for key in ['iou', 'dice', 'cldice', 'vesselness']:
                values = [s.get(key) for s in steps if key in s]
                if values:
                    episode_summary[f'avg_{key}'] = np.mean(values)
                    episode_summary[f'final_{key}'] = values[-1] if values else None
        
        self.episode_metrics.append(episode_summary)
        
        return episode_summary
    
    def log_sam2_metrics(self, metrics: Dict[str, Any]):
        """记录 SAM2 相关指标"""
        metrics['timestamp'] = time.time()
        self.sam2_metrics.append(metrics)
    
    def log_timing(self, operation: str, duration: float, details: Dict = None):
        """记录时间统计"""
        timing_data = {
            'operation': operation,
            'duration': duration,
            'timestamp': time.time()
        }
        if details:
            timing_data.update(details)
        self.timing_metrics.append(timing_data)
    
    def _convert_to_serializable(self, obj):
        """转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
    
    def save(self):
        """保存所有指标到文件"""
        # 转换numpy类型
        episode_metrics = self._convert_to_serializable(self.episode_metrics)
        step_metrics = self._convert_to_serializable(self.step_metrics)
        sam2_metrics = self._convert_to_serializable(self.sam2_metrics)
        timing_metrics = self._convert_to_serializable(self.timing_metrics)
        
        # 保存原始数据
        with open(self.log_dir / 'episode_metrics.json', 'w') as f:
            json.dump(episode_metrics, f, indent=2)
        
        with open(self.log_dir / 'step_metrics.json', 'w') as f:
            json.dump(step_metrics, f, indent=2)
        
        with open(self.log_dir / 'sam2_metrics.json', 'w') as f:
            json.dump(sam2_metrics, f, indent=2)
        
        with open(self.log_dir / 'timing_metrics.json', 'w') as f:
            json.dump(timing_metrics, f, indent=2)
        
        # 生成汇总报告
        self.generate_summary_report()
        
        print(f"✓ 指标已保存到: {self.log_dir}")
    
    def generate_summary_report(self):
        """生成汇总报告"""
        report_path = self.log_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("训练/推理汇总报告\n")
            f.write("=" * 60 + "\n\n")
            
            # Episode 统计
            if self.episode_metrics:
                f.write("【Episode 统计】\n")
                f.write(f"  总 Episode 数: {len(self.episode_metrics)}\n")
                
                avg_reward = np.mean([e['total_reward'] for e in self.episode_metrics])
                avg_steps = np.mean([e['total_steps'] for e in self.episode_metrics])
                avg_duration = np.mean([e['duration'] for e in self.episode_metrics])
                
                f.write(f"  平均总奖励: {avg_reward:.4f}\n")
                f.write(f"  平均步数: {avg_steps:.2f}\n")
                f.write(f"  平均时长: {avg_duration:.2f} 秒\n\n")
                
                # 最终性能指标
                f.write("【性能指标】\n")
                for metric in ['iou', 'dice', 'cldice', 'vesselness']:
                    final_key = f'final_{metric}'
                    values = [e.get(final_key) for e in self.episode_metrics if final_key in e]
                    if values:
                        f.write(f"  最终 {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
                f.write("\n")
            
            # 动作统计
            if self.step_metrics:
                f.write("【动作统计】\n")
                action_counts = {}
                for step in self.step_metrics:
                    action_name = step.get('action_name', 'unknown')
                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
                
                total_actions = len(self.step_metrics)
                for action_name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
                    f.write(f"  {action_name}: {count} ({count/total_actions*100:.1f}%)\n")
                f.write("\n")
            
            # 时间统计
            if self.timing_metrics:
                f.write("【时间统计】\n")
                timing_by_op = {}
                for t in self.timing_metrics:
                    op = t['operation']
                    if op not in timing_by_op:
                        timing_by_op[op] = []
                    timing_by_op[op].append(t['duration'])
                
                for op, durations in sorted(timing_by_op.items()):
                    f.write(f"  {op}: {np.mean(durations)*1000:.2f} ms (平均)\n")
                f.write("\n")
            
            # SAM2 统计
            if self.sam2_metrics:
                f.write("【SAM2 统计】\n")
                total_candidates = sum(m.get('num_candidates', 0) for m in self.sam2_metrics)
                f.write(f"  总生成候选数: {total_candidates}\n")
                f.write(f"  平均每次生成: {total_candidates/len(self.sam2_metrics):.2f}\n")
        
        print(f"✓ 汇总报告已生成: {report_path}")
    
    def plot_metrics(self):
        """绘制指标曲线"""
        if not self.episode_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 奖励曲线
        episodes = [e['episode_id'] for e in self.episode_metrics]
        rewards = [e['total_reward'] for e in self.episode_metrics]
        axes[0, 0].plot(episodes, rewards, marker='o', markersize=3)
        axes[0, 0].set_title('Total Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # 2. 步数曲线
        steps = [e['total_steps'] for e in self.episode_metrics]
        axes[0, 1].plot(episodes, steps, marker='o', markersize=3, color='orange')
        axes[0, 1].set_title('Steps per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # 3. 性能指标（IoU/Dice/clDice）
        for metric, color in [('iou', 'green'), ('dice', 'blue'), ('cldice', 'red')]:
            final_key = f'final_{metric}'
            values = [e.get(final_key) for e in self.episode_metrics if final_key in e]
            if values:
                axes[1, 0].plot(range(len(values)), values, 
                               marker='o', markersize=3, label=metric, color=color)
        axes[1, 0].set_title('Segmentation Metrics')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 动作分布
        action_counts = {}
        for step in self.step_metrics:
            action_name = step.get('action_name', 'unknown')
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        axes[1, 1].bar(actions, counts)
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].set_xlabel('Action')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = self.log_dir / 'metrics_plot.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"✓ 指标图表已生成: {plot_path}")

