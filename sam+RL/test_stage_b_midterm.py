"""
阶段B中期测试：10000步训练
验证长期训练的收敛性和性能提升
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
    
    # 修改为中期测试配置
    config['training']['total_timesteps'] = 10000
    config['training']['log_dir'] = './logs/stage_b_midterm'
    config['training']['save_dir'] = './checkpoints/stage_b_midterm'
    config['training']['save_freq'] = 2000  # 每2000步保存一次
    
    # 保存临时配置
    temp_config_path = '/tmp/stage_b_midterm_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建参数对象
    class Args:
        config = temp_config_path
    
    print(f"\n{'='*80}")
    print(f"阶段B中期测试：10000步训练")
    print(f"\n初步测试结果（2048步）:")
    print(f"  - 平均IoU: 4.59%")
    print(f"  - 最佳IoU: 73.40%")
    print(f"  - 平均奖励: +1.04")
    print(f"\n中期目标（10000步）:")
    print(f"  - 平均IoU: 10-15%")
    print(f"  - 最佳IoU: 80-90%")
    print(f"  - 平均奖励: +2.0+")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    # 获取最终指标
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"阶段B中期测试完成")
    print(f"{'='*80}")
    print(f"\n最终结果:")
    print(f"  - 平均IoU: {summary.get('avg_final_iou', 0)*100:.2f}%")
    print(f"  - 最佳IoU: {summary.get('best_iou', 0)*100:.2f}%")
    print(f"  - 平均奖励: {summary.get('avg_episode_reward', 0):.4f}")
    print(f"\n是否达到目标:")
    avg_iou = summary.get('avg_final_iou', 0) * 100
    best_iou = summary.get('best_iou', 0) * 100
    
    if avg_iou >= 10:
        print(f"  ✅ 平均IoU达标: {avg_iou:.2f}% >= 10%")
    else:
        print(f"  ⚠️  平均IoU未达标: {avg_iou:.2f}% < 10%")
    
    if best_iou >= 80:
        print(f"  ✅ 最佳IoU达标: {best_iou:.2f}% >= 80%")
    else:
        print(f"  ⚠️  最佳IoU未达标: {best_iou:.2f}% < 80%")
    
    print(f"{'='*80}\n")

