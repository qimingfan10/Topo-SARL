#!/usr/bin/env python3
"""
修复RewardNet：从Lightning checkpoint中提取模型权重，或重新训练
"""

import os
import sys
from pathlib import Path
import torch
from glob import glob

def find_latest_checkpoint(base_dir):
    """查找最新的checkpoint文件"""
    ckpt_files = glob(f"{base_dir}/Reward3DOptimizer_test/version_*/checkpoints/*.ckpt")
    if not ckpt_files:
        return None
    # 按修改时间排序，返回最新的
    ckpt_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return ckpt_files[0]

def extract_and_save_model(ckpt_path, output_path):
    """从Lightning checkpoint中提取模型权重并保存"""
    print(f"正在加载checkpoint: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Lightning checkpoint包含完整的state_dict，我们只需要net的权重
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 提取net的权重（去掉'net.'前缀）
            net_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('net.'):
                    new_key = key[4:]  # 去掉'net.'前缀
                    net_state_dict[new_key] = value
            
            if net_state_dict:
                # 保存净化的权重
                torch.save(net_state_dict, output_path)
                print(f"✓ 成功保存RewardNet模型到: {output_path}")
                print(f"  模型包含 {len(net_state_dict)} 个参数")
                return True
            else:
                print("✗ checkpoint中没有找到net的权重")
                return False
        else:
            print("✗ checkpoint格式不正确")
            return False
    except Exception as e:
        print(f"✗ 提取失败: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="修复RewardNet模型")
    parser.add_argument('--log-dir', default='/home/ubuntu/my_rl4seg3d_logs/3d_test',
                       help='日志目录路径')
    parser.add_argument('--iteration', type=int, default=0,
                       help='要保存到哪个迭代目录 (默认: 0)')
    parser.add_argument('--checkpoint', default=None,
                       help='指定checkpoint文件路径（可选）')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    # 查找checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt_path = args.checkpoint
    else:
        print(f"在 {log_dir} 中搜索checkpoint...")
        ckpt_path = find_latest_checkpoint(str(log_dir))
        
        if not ckpt_path:
            print("\n✗ 没有找到任何checkpoint文件")
            print("\n建议操作：")
            print("1. 重新运行训练以生成和保存RewardNet模型")
            print("2. 或者检查训练日志，确保训练成功完成")
            print("\n重新训练命令：")
            print(f"  cd /home/ubuntu/RL4Seg3D")
            print(f"  python3 rl4seg3d/auto_iteration.py")
            return 1
    
    print(f"\n找到checkpoint: {ckpt_path}")
    
    # 创建输出目录
    output_dir = log_dir / str(args.iteration)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rewardnet.ckpt"
    
    # 提取并保存
    if extract_and_save_model(ckpt_path, str(output_path)):
        print(f"\n✓ 修复成功！")
        print(f"RewardNet模型已保存到: {output_path}")
        
        # 验证文件
        file_size = Path(output_path).stat().st_size / (1024*1024)
        print(f"文件大小: {file_size:.2f} MB")
        
        # 提示用户可以继续训练
        print(f"\n现在可以继续训练了！下次迭代将使用这个RewardNet模型。")
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())

