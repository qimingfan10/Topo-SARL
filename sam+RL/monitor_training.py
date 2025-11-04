"""
训练监控脚本
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta


def parse_log_file(log_file):
    """解析训练日志"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    info = {
        'total_lines': len(lines),
        'status': 'unknown',
        'episodes': 0,
        'merges': 0,
        'samples': 0,
        'last_10_lines': lines[-10:] if len(lines) >= 10 else lines
    }
    
    # 检查状态
    if any('训练完成' in line for line in lines):
        info['status'] = 'completed'
    elif any('Error' in line or 'Traceback' in line for line in lines):
        info['status'] = 'error'
    elif any('开始训练' in line for line in lines):
        info['status'] = 'running'
    else:
        info['status'] = 'starting'
    
    # 统计信息
    for line in lines:
        if '生成候选数' in line:
            info['samples'] += 1
        if '[MERGE]' in line and '选中候选数' in line:
            info['merges'] += 1
    
    # 估算episodes（每个episode大约有多个候选生成）
    info['episodes'] = info['samples'] // 3  # 粗略估计
    
    return info


def find_checkpoints(checkpoint_dir):
    """查找保存的检查点"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.zip'):
            path = os.path.join(checkpoint_dir, f)
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            mtime = os.path.getmtime(path)
            checkpoints.append({
                'name': f,
                'size_mb': size,
                'modified': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)


def find_tensorboard_logs(log_dir):
    """查找TensorBoard日志"""
    if not os.path.exists(log_dir):
        return []
    
    logs = []
    for root, dirs, files in os.walk(log_dir):
        for d in dirs:
            if d.startswith('PPO_'):
                path = os.path.join(root, d)
                logs.append({
                    'name': d,
                    'path': path
                })
    
    return logs


def display_status(log_file, checkpoint_dir, log_dir):
    """显示训练状态"""
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 80)
    print(" " * 25 + "SAM2 + RL 训练监控")
    print("=" * 80)
    print()
    
    # 当前时间
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 解析日志
    info = parse_log_file(log_file)
    
    if info is None:
        print("⚠️  日志文件不存在")
        print(f"   路径: {log_file}")
    else:
        # 训练状态
        status_emoji = {
            'running': '🟢',
            'completed': '✅',
            'error': '❌',
            'starting': '🟡',
            'unknown': '⚪'
        }
        
        print(f"训练状态: {status_emoji.get(info['status'], '?')} {info['status'].upper()}")
        print()
        
        # 统计信息
        print("训练进度:")
        print(f"  Episodes (估计): {info['episodes']}")
        print(f"  候选生成次数: {info['samples']}")
        print(f"  Merge执行次数: {info['merges']}")
        print(f"  日志行数: {info['total_lines']}")
        print()
        
        # 最后几行
        print("最后10行日志:")
        print("-" * 80)
        for line in info['last_10_lines']:
            print(line.rstrip())
        print("-" * 80)
        print()
    
    # 检查点
    checkpoints = find_checkpoints(checkpoint_dir)
    if checkpoints:
        print(f"已保存的检查点 ({len(checkpoints)}):")
        for cp in checkpoints[:5]:  # 只显示最新的5个
            print(f"  {cp['name']} | {cp['size_mb']:.1f} MB | {cp['modified']}")
    else:
        print("已保存的检查点: 0")
    print()
    
    # TensorBoard日志
    tb_logs = find_tensorboard_logs(log_dir)
    if tb_logs:
        print(f"TensorBoard日志 ({len(tb_logs)}):")
        for log in tb_logs:
            print(f"  {log['name']} | {log['path']}")
        print()
        print("查看TensorBoard:")
        print(f"  tensorboard --logdir {log_dir} --port 6006")
    else:
        print("TensorBoard日志: 无")
    print()
    
    # 帮助信息
    print("=" * 80)
    print("监控命令:")
    print(f"  查看完整日志: tail -f {log_file}")
    print(f"  停止训练: pkill -f 'python3 train.py'")
    print(f"  查看进程: ps aux | grep train.py")
    print()
    print("按 Ctrl+C 退出监控")
    print("=" * 80)


def main():
    """主函数"""
    log_file = '/home/ubuntu/sam+RL/logs/full_training_v2.log'
    checkpoint_dir = '/home/ubuntu/sam+RL/checkpoints'
    log_dir = '/home/ubuntu/sam+RL/logs'
    
    print("启动训练监控...")
    print(f"日志文件: {log_file}")
    print()
    print("每5秒更新一次，按 Ctrl+C 退出")
    time.sleep(2)
    
    try:
        while True:
            display_status(log_file, checkpoint_dir, log_dir)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n监控已停止")


if __name__ == "__main__":
    main()

