#!/usr/bin/env python3
"""
在真实血管造影数据集上测试阶段B方法
使用成功的opt2配置
"""
import sys
import yaml

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    config_path = '/home/ubuntu/sam+RL/config/stage_b_opt2.yaml'
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 调整为快速测试（5000步验证）
    config['training']['total_timesteps'] = 5000
    config['training']['save_freq'] = 2000
    config['training']['log_dir'] = './logs/stage_b_real_dataset'
    config['training']['save_dir'] = './checkpoints/stage_b_real_dataset'
    
    # 保存临时配置
    temp_config_path = '/tmp/stage_b_real_dataset_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    class Args:
        config = temp_config_path
    
    print(f"\n{'='*80}")
    print(f"🔬 真实数据集测试（5000步）")
    print(f"{'='*80}")
    print(f"\n数据集信息:")
    print(f"  ✅ 图像数量: 220个")
    print(f"  ✅ 数据类型: 血管造影图像")
    print(f"  ✅ 掩膜覆盖率: 平均3.36%")
    print(f"  ✅ 图像尺寸: 512×512")
    print(f"\n使用配置:")
    print(f"  📋 成功的opt2配置（18.35% IoU）")
    print(f"  🔧 环境强制最小步数: 5步")
    print(f"  🎯 训练步数: 5000步（快速验证）")
    print(f"\n对比:")
    print(f"  之前临时数据: 3个SAM2示例图像")
    print(f"  现在真实数据: 220个血管造影图像")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"🏁 真实数据集测试完成")
    print(f"{'='*80}")
    
    real_iou = f'{summary.get("avg_final_iou", 0)*100:.2f}%'
    real_best = f'{summary.get("best_iou", 0)*100:.2f}%'
    real_len = f'{summary.get("avg_episode_length", 0):.2f}步'
    real_reward = f'{summary.get("avg_episode_reward", 0):+.2f}'
    
    print(f"\n结果对比:")
    print(f"┌{'─'*40}┬{'─'*15}┬{'─'*15}┐")
    print(f"│ {'指标':<38} │ {'临时数据(3图)':<13} │ {'真实数据(220图)':<13} │")
    print(f"├{'─'*40}┼{'─'*15}┼{'─'*15}┤")
    print(f"│ {'平均IoU':<38} │ {'18.35%':<13} │ {real_iou:<13} │")
    print(f"│ {'最佳IoU':<38} │ {'43.73%':<13} │ {real_best:<13} │")
    print(f"│ {'Episode长度':<38} │ {'6.93步':<13} │ {real_len:<13} │")
    print(f"│ {'平均奖励':<38} │ {'+3.78':<13} │ {real_reward:<13} │")
    print(f"│ {'图像数量':<38} │ {'3':<13} │ {'220':<13} │")
    print(f"│ {'数据类型':<38} │ {'示例图像':<13} │ {'血管造影':<13} │")
    print(f"└{'─'*40}┴{'─'*15}┴{'─'*15}┘")
    
    # 评估
    avg_iou = summary.get('avg_final_iou', 0) * 100
    
    print(f"\n📊 真实数据集上的表现:")
    if avg_iou >= 10.0:
        print(f"  ✅ 优秀！平均IoU: {avg_iou:.2f}%")
        print(f"  方法在真实血管分割任务上表现良好")
    elif avg_iou >= 5.0:
        print(f"  ✅ 良好！平均IoU: {avg_iou:.2f}%")
        print(f"  考虑到血管很细（平均占3.36%），这是合理结果")
    elif avg_iou >= 2.0:
        print(f"  ⚠️  可接受：{avg_iou:.2f}%")
        print(f"  血管分割确实很困难，可能需要更多训练")
    else:
        print(f"  ⚠️  需要改进：{avg_iou:.2f}%")
        print(f"  建议：增加训练步数或调整超参数")
    
    # 与阶段A对比
    stage_a_iou = 1.08
    if avg_iou > stage_a_iou:
        improvement = avg_iou / stage_a_iou
        print(f"\n🏆 相比阶段A:")
        print(f"  提升倍数: {improvement:.1f}x")
        print(f"  绝对提升: +{avg_iou - stage_a_iou:.2f}%")
        if improvement >= 5:
            print(f"  ✅ 显著优于阶段A！")
        else:
            print(f"  ✅ 优于阶段A")
    
    print(f"\n💡 建议:")
    if avg_iou >= 5.0:
        print(f"  1. 可以进行长期训练（20000-50000步）")
        print(f"  2. 期望平均IoU可达 {avg_iou * 2:.1f}% - {avg_iou * 3:.1f}%")
    else:
        print(f"  1. 继续训练10000-20000步")
        print(f"  2. 可以尝试调整学习率或奖励权重")
    
    print(f"\n{'='*80}\n")

