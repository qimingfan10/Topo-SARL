"""
基线测试脚本：1024步训练，收集性能指标
"""
import sys
import yaml
import argparse

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_enhanced import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基线测试')
    parser.add_argument(
        '--steps',
        type=int,
        default=1024,
        help='训练步数'
    )
    
    args_parsed = parser.parse_args()
    
    # 加载配置
    config_path = '/home/ubuntu/sam+RL/config/default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改训练步数
    config['training']['total_timesteps'] = args_parsed.steps
    config['training']['log_dir'] = './logs/baseline_test'
    config['training']['save_dir'] = './checkpoints/baseline_test'
    
    # 保存临时配置
    temp_config_path = '/tmp/baseline_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建参数对象
    class Args:
        config = temp_config_path
    
    print(f"\n{'='*80}")
    print(f"基线测试：{args_parsed.steps}步训练")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    print(f"\n{'='*80}")
    print(f"基线测试完成")
    print(f"{'='*80}\n")

