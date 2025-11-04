"""
阶段B初步训练测试：1024步快速验证
"""
import sys
import yaml

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    # 加载配置
    config_path = '/home/ubuntu/sam+RL/config/stage_b.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改为快速测试配置
    config['training']['total_timesteps'] = 2048  # 快速测试
    config['training']['log_dir'] = './logs/stage_b_test'
    config['training']['save_dir'] = './checkpoints/stage_b_test'
    
    # 保存临时配置
    temp_config_path = '/tmp/stage_b_test_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建参数对象
    class Args:
        config = temp_config_path
    
    print(f"\n{'='*80}")
    print(f"阶段B初步测试：2048步训练")
    print(f"预期：IoU应该比阶段A（1%）有显著提升")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    print(f"\n{'='*80}")
    print(f"阶段B初步测试完成")
    print(f"\n对比阶段A基线:")
    print(f"  - 阶段A平均IoU: 1.08%")
    print(f"  - 阶段A最佳IoU: 32.84%")
    print(f"\n阶段B结果（上方）应该显示改进！")
    print(f"{'='*80}\n")

