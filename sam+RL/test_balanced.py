"""
平衡版测试脚本：适度的假阳性惩罚（-1.0）
"""
import sys
import yaml

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_enhanced import train


if __name__ == "__main__":
    # 加载平衡配置
    config_path = '/home/ubuntu/sam+RL/config/balanced.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改训练步数
    config['training']['total_timesteps'] = 2048
    
    # 保存临时配置
    temp_config_path = '/tmp/balanced_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建参数对象
    class Args:
        config = temp_config_path
    
    print(f"\n{'='*80}")
    print(f"平衡版测试：2048步训练")
    print(f"使用平衡配置：IoU权重3.0，假阳性惩罚-1.0（适度）")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    print(f"\n{'='*80}")
    print(f"平衡版测试完成")
    print(f"查看对比：基线(-0.22) vs 改进v1(-0.99) vs 平衡v2(?)")
    print(f"{'='*80}\n")

