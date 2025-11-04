"""
优化循环1：测试新奖励函数
目标：验证新奖励是否能增加Episode长度并提升IoU
"""
import sys
import yaml

sys.path.insert(0, '/home/ubuntu/sam+RL')

from train_stage_b import train


if __name__ == "__main__":
    # 加载优化配置
    config_path = '/home/ubuntu/sam+RL/config/stage_b_optimized.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改为快速测试配置
    config['training']['total_timesteps'] = 5000  # 5000步快速验证
    config['training']['log_dir'] = './logs/stage_b_opt1'
    config['training']['save_dir'] = './checkpoints/stage_b_opt1'
    config['training']['save_freq'] = 2000
    
    # 保存临时配置
    temp_config_path = '/tmp/stage_b_opt1_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建参数对象
    class Args:
        config = temp_config_path
    
    print(f"\n{'='*80}")
    print(f"优化循环1：测试新奖励函数（5000步）")
    print(f"{'='*80}")
    print(f"\n关键改进:")
    print(f"  1. 减小Bonus权重：从2.0/1.0/0.5 → 0.15/0.09/0.03（减少90%）")
    print(f"  2. 增加最小步数要求：min_steps=5")
    print(f"  3. 添加探索奖励：+0.05/步（超过5步后）")
    print(f"  4. 过早终止惩罚：-0.5（< 5步terminate）")
    print(f"  5. 降低学习率：0.0003 → 0.0001（提高稳定性）")
    print(f"  6. 提高探索：ent_coef从0.1 → 0.2")
    print(f"\n预期效果:")
    print(f"  - Episode长度增加：从2.9步 → 5-8步")
    print(f"  - 平均IoU提升：从4.3% → 6-8%")
    print(f"  - 训练更稳定：IoU不再下降")
    print(f"{'='*80}\n")
    
    # 运行训练
    metrics_tracker = train(Args())
    
    # 获取结果
    summary = metrics_tracker.get_summary(last_n=100)
    
    print(f"\n{'='*80}")
    print(f"优化循环1完成")
    print(f"{'='*80}")
    print(f"\n结果对比:")
    opt1_iou = f'{summary.get("avg_final_iou", 0)*100:.2f}%'
    opt1_best = f'{summary.get("best_iou", 0)*100:.2f}%'
    opt1_len = f'{summary.get("avg_episode_length", 0):.2f}步'
    opt1_reward = f'{summary.get("avg_episode_reward", 0):+.2f}'
    
    print(f"┌{'─'*40}┬{'─'*12}┬{'─'*12}┬{'─'*12}┐")
    print(f"│ {'指标':<38} │ {'初步测试':<10} │ {'中期测试':<10} │ {'优化1':<10} │")
    print(f"├{'─'*40}┼{'─'*12}┼{'─'*12}┼{'─'*12}┤")
    print(f"│ {'平均IoU':<38} │ {'4.59%':<10} │ {'4.30%':<10} │ {opt1_iou:<10} │")
    print(f"│ {'最佳IoU':<38} │ {'73.40%':<10} │ {'75.49%':<10} │ {opt1_best:<10} │")
    print(f"│ {'Episode长度':<38} │ {'2.79步':<10} │ {'2.93步':<10} │ {opt1_len:<10} │")
    print(f"│ {'平均奖励':<38} │ {'+1.04':<10} │ {'+0.55':<10} │ {opt1_reward:<10} │")
    print(f"└{'─'*40}┴{'─'*12}┴{'─'*12}┴{'─'*12}┘")
    
    # 评估是否改善
    avg_iou = summary.get('avg_final_iou', 0) * 100
    avg_len = summary.get('avg_episode_length', 0)
    
    print(f"\n评估:")
    improvements = []
    concerns = []
    
    if avg_iou > 4.59:
        improvements.append(f"  ✅ IoU提升: {avg_iou:.2f}% > 4.59%")
    else:
        concerns.append(f"  ⚠️  IoU未提升: {avg_iou:.2f}% ≤ 4.59%")
    
    if avg_len > 3.5:
        improvements.append(f"  ✅ Episode长度增加: {avg_len:.2f}步 > 3步")
    else:
        concerns.append(f"  ⚠️  Episode长度仍短: {avg_len:.2f}步")
    
    if avg_iou >= 6.0:
        improvements.append(f"  🎉 达到6%目标！")
    
    if improvements:
        print("\n改善点:")
        for imp in improvements:
            print(imp)
    
    if concerns:
        print("\n需关注:")
        for con in concerns:
            print(con)
    
    # 下一步建议
    print(f"\n{'='*80}")
    if avg_iou >= 6.0:
        print("✅ 成功！继续长期训练争取达到10%")
        print("建议：运行20000-50000步训练")
    elif avg_iou > 4.8:
        print("⚠️  有改善但不够，需要进一步调整")
        print("建议：")
        print("  1. 增加min_steps到7-10步")
        print("  2. 增大exploration_bonus")
        print("  3. 或尝试更大的网格（64×64）")
    else:
        print("❌ 改善不明显，需要重新思考策略")
        print("建议：")
        print("  1. 分析失败样本，找出共同问题")
        print("  2. 可能需要课程学习")
        print("  3. 或考虑添加其他奖励信号")
    print(f"{'='*80}\n")

