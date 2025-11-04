#!/usr/bin/env python3
"""
RewardNet诊断脚本 - 快速检查RewardNet状态和数据
"""

import sys
from pathlib import Path

def check_rewardnet_checkpoints(log_dir):
    """检查RewardNet checkpoint文件"""
    print("=" * 60)
    print("检查 RewardNet Checkpoint 文件")
    print("=" * 60)
    
    log_path = Path(log_dir)
    found_any = False
    
    for i in range(10):
        ckpt_path = log_path / str(i) / "rewardnet.ckpt"
        if ckpt_path.exists():
            import torch
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                size_mb = ckpt_path.stat().st_size / (1024*1024)
                print(f"✓ 迭代{i}: {ckpt_path}")
                print(f"  大小: {size_mb:.2f} MB, 参数数量: {len(ckpt)}")
                found_any = True
            except Exception as e:
                print(f"✗ 迭代{i}: 文件存在但无法加载: {e}")
        else:
            if i < 5:  # 只显示前5个
                print(f"✗ 迭代{i}: 未找到 {ckpt_path}")
    
    if not found_any:
        print("\n⚠️  没有找到任何RewardNet checkpoint!")
        print("这意味着RewardNet从未被保存，将使用随机初始化的网络。")
    
    return found_any

def check_reward_training_data(log_dir):
    """检查RewardNet训练数据"""
    print("\n" + "=" * 60)
    print("检查 RewardNet 训练数据 (rewardDS)")
    print("=" * 60)
    
    reward_ds = Path(log_dir) / "rewardDS"
    
    if not reward_ds.exists():
        print(f"✗ rewardDS目录不存在: {reward_ds}")
        print("\n说明：需要先运行预测生成训练数据")
        return False
    
    print(f"✓ rewardDS目录存在: {reward_ds}\n")
    
    # 检查子目录
    for subdir in ['images', 'gt', 'pred']:
        subdir_path = reward_ds / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.nii.gz"))
            print(f"✓ {subdir}/: {len(files)} 个文件")
            if files and len(files) <= 5:
                for f in files[:5]:
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"    - {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {subdir}/: 目录不存在")
            return False
    
    # 评估数据量
    num_samples = len(list((reward_ds / "images").glob("*.nii.gz")))
    print(f"\n总样本数: {num_samples}")
    
    if num_samples < 2:
        print("❌ 样本太少！无法训练RewardNet（至少需要2个）")
        return False
    elif num_samples < 10:
        print("⚠️  样本较少（<10个），建议增加预测数量生成更多数据")
        print("   修改config/auto_iteration.yaml中的 rl_num_predict")
        return True
    else:
        print("✓ 样本数量充足")
        return True

def check_reward_training_logs(log_dir):
    """检查RewardNet训练日志"""
    print("\n" + "=" * 60)
    print("检查 RewardNet 训练日志")
    print("=" * 60)
    
    reward_log_dir = Path(log_dir) / "Reward3DOptimizer_test"
    
    if not reward_log_dir.exists():
        print(f"✗ 训练日志目录不存在: {reward_log_dir}")
        print("说明：RewardNet从未被训练过")
        return False
    
    versions = list(reward_log_dir.glob("version_*"))
    if not versions:
        print("✗ 没有找到任何训练版本")
        return False
    
    print(f"✓ 找到 {len(versions)} 个训练版本")
    print(f"最新版本: {max(versions, key=lambda x: int(x.name.split('_')[1])).name}")
    
    # 检查是否有checkpoint
    has_ckpt = False
    for v in versions[-5:]:  # 检查最近5个
        ckpt_dir = v / "checkpoints"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                print(f"  {v.name}: 有checkpoint ({len(ckpts)}个)")
                has_ckpt = True
    
    if not has_ckpt:
        print("\n⚠️  训练日志存在但没有checkpoint文件")
        print("原因：训练完成后save_model()没有被调用（已在新代码中修复）")
    
    return has_ckpt

def provide_recommendations(has_ckpt, has_data):
    """提供修复建议"""
    print("\n" + "=" * 60)
    print("诊断结果和建议")
    print("=" * 60)
    
    if has_ckpt:
        print("✓ 状态良好！RewardNet checkpoint已存在。")
        print("\n可以继续训练：")
        print("  cd /home/ubuntu/RL4Seg3D")
        print("  python3 rl4seg3d/auto_iteration.py")
    elif not has_data:
        print("❌ 缺少训练数据！")
        print("\n解决方案：运行完整流程生成数据")
        print("  cd /home/ubuntu/RL4Seg3D")
        print("  python3 rl4seg3d/auto_iteration.py")
    else:
        print("⚠️  有数据但没有训练好的RewardNet模型")
        print("\n建议操作：")
        print("\n方案1（推荐）：重新运行完整训练流程")
        print("  cd /home/ubuntu/RL4Seg3D")
        print("  python3 rl4seg3d/auto_iteration.py")
        print("\n方案2：单独训练RewardNet")
        print("  cd /home/ubuntu/RL4Seg3D")
        print("  python3 scripts/train_rewardnet_standalone.py \\")
        print("      --data-dir /home/ubuntu/my_rl4seg3d_logs/3d_test \\")
        print("      --epochs 50")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="诊断RewardNet状态")
    parser.add_argument('--log-dir', default='/home/ubuntu/my_rl4seg3d_logs/3d_test',
                       help='日志目录路径')
    args = parser.parse_args()
    
    print("\n🔍 RewardNet 诊断工具\n")
    
    has_ckpt = check_rewardnet_checkpoints(args.log_dir)
    has_data = check_reward_training_data(args.log_dir)
    check_reward_training_logs(args.log_dir)
    provide_recommendations(has_ckpt, has_data)
    
    print("\n" + "=" * 60)
    print("诊断完成！")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()

