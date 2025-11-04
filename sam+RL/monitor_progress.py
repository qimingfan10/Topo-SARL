"""
监控训练进度
"""
import json
import time
from pathlib import Path
import sys


def monitor_training(log_dir='./logs/baseline_test', interval=10):
    """
    监控训练进度
    
    Args:
        log_dir: 日志目录
        interval: 检查间隔（秒）
    """
    metrics_file = Path(log_dir) / 'metrics.json'
    
    print(f"监控训练进度: {log_dir}")
    print(f"指标文件: {metrics_file}")
    print(f"检查间隔: {interval}秒")
    print("=" * 80)
    print()
    
    last_episodes = 0
    last_steps = 0
    
    while True:
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                total_steps = data.get('total_steps', 0)
                total_episodes = data.get('total_episodes', 0)
                episode_rewards = data.get('episode_rewards', [])
                episode_ious = data.get('episode_final_ious', [])
                
                # 计算进度
                new_episodes = total_episodes - last_episodes
                new_steps = total_steps - last_steps
                
                # 最近的统计
                if len(episode_rewards) > 0:
                    recent_rewards = episode_rewards[-10:]
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                else:
                    avg_reward = 0
                
                if len(episode_ious) > 0:
                    recent_ious = episode_ious[-10:]
                    avg_iou = sum(recent_ious) / len(recent_ious)
                else:
                    avg_iou = 0
                
                # 打印进度
                print(f"\r[{time.strftime('%H:%M:%S')}] "
                      f"步数: {total_steps:5d} (+{new_steps:3d}) | "
                      f"Episodes: {total_episodes:4d} (+{new_episodes:2d}) | "
                      f"平均奖励: {avg_reward:7.4f} | "
                      f"平均IoU: {avg_iou:6.4f}", 
                      end='', flush=True)
                
                last_episodes = total_episodes
                last_steps = total_steps
                
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] 等待指标文件...", end='', flush=True)
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else './logs/baseline_test'
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    monitor_training(log_dir, interval)

