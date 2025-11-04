"""
改进测试脚本：使用优化的配置
"""
import sys
import yaml
import argparse

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_enhanced import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='改进测试')
    parser.add_argument(
        '--steps',
        type=int,
        default=1024,
        help='训练步数'
    )
    
    args_parsed = parser.parse_args()
    
    # 加载优化配置
    config_path = '/home/ubuntu/sam+RL/config/improved.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改训练步数
    config['training']['total_timesteps'] = args_parsed.steps
    
    # 保存临时配置
    temp_config_path = '/tmp/improved_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建参数对象
    class Args:
        config = temp_config_path
    
    print(f"\n{'='*80}")
    print(f"改进测试：{args_parsed.steps}步训练")
    print(f"使用优化配置：IoU权重5.0，假阳性惩罚-2.0")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    print(f"\n{'='*80}")
    print(f"改进测试完成")
    print(f"{'='*80}\n")

