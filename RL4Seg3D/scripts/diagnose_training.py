#!/usr/bin/env python3
"""
训练状态诊断工具
检查配置、数据、checkpoint等是否正常
"""

import argparse
import yaml
from pathlib import Path
import sys


def diagnose_config(config_path):
    """诊断配置文件"""
    print("\n" + "="*70)
    print("1️⃣  配置文件诊断")
    print("="*70)
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    issues = []
    suggestions = []
    
    # 检查训练时长
    num_iter = config.get('num_iter', 0)
    rl_num_epochs = config.get('rl_num_epochs', 0)
    
    print(f"\n📊 训练时长:")
    print(f"   num_iter:       {num_iter}")
    print(f"   rl_num_epochs:  {rl_num_epochs}")
    
    if num_iter < 30:
        issues.append(f"num_iter太小 ({num_iter})")
        suggestions.append(f"建议: 将num_iter增加到至少30-50")
        print(f"   ⚠️  num_iter={num_iter} 可能不够，建议>=30")
    else:
        print(f"   ✅ num_iter充足")
    
    if rl_num_epochs < 30:
        issues.append(f"rl_num_epochs太小 ({rl_num_epochs})")
        suggestions.append(f"建议: 将rl_num_epochs增加到至少30-50")
        print(f"   ⚠️  rl_num_epochs={rl_num_epochs} 可能不够，建议>=30")
    else:
        print(f"   ✅ rl_num_epochs充足")
    
    # 检查RewardNet配置
    rl_num_predict = config.get('rl_num_predict', 0)
    print(f"\n🎯 RewardNet配置:")
    print(f"   rl_num_predict: {rl_num_predict}")
    
    if rl_num_predict < 1000:
        issues.append(f"rl_num_predict太小 ({rl_num_predict})")
        suggestions.append(f"建议: 将rl_num_predict增加到至少3000-5000")
        print(f"   ⚠️  rl_num_predict={rl_num_predict} 可能不够，建议>=3000")
    else:
        print(f"   ✅ rl_num_predict充足")
    
    # 检查学习率
    if 'model' in config:
        actor_lr = config['model'].get('actor_lr', 'N/A')
        critic_lr = config['model'].get('critic_lr', 'N/A')
        print(f"\n📈 学习率:")
        print(f"   actor_lr:  {actor_lr}")
        print(f"   critic_lr: {critic_lr}")
        
        if isinstance(actor_lr, float) and actor_lr > 0.0005:
            issues.append(f"actor_lr可能过大 ({actor_lr})")
            suggestions.append(f"建议: 降低actor_lr到0.0001-0.0003")
            print(f"   ⚠️  actor_lr可能过大，建议<=0.0003")
    
    return len(issues) == 0, issues, suggestions


def diagnose_checkpoints(log_dir):
    """诊断checkpoint状态"""
    print("\n" + "="*70)
    print("2️⃣  Checkpoint诊断")
    print("="*70)
    
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return False
    
    # 查找所有checkpoint
    ckpts = list(log_dir.rglob("*.ckpt"))
    print(f"\n📦 找到 {len(ckpts)} 个checkpoint文件")
    
    if len(ckpts) == 0:
        print(f"   ⚠️  没有找到checkpoint，可能还没开始训练")
        return False
    
    # 分类checkpoint
    best_ckpts = [c for c in ckpts if 'best' in c.name.lower()]
    last_ckpts = [c for c in ckpts if 'last' in c.name.lower()]
    actor_ckpts = [c for c in ckpts if 'actor' in c.name.lower()]
    reward_ckpts = [c for c in ckpts if 'reward' in c.name.lower()]
    
    print(f"\n分类统计:")
    print(f"   best.ckpt:   {len(best_ckpts)} 个")
    print(f"   last.ckpt:   {len(last_ckpts)} 个")
    print(f"   actor.ckpt:  {len(actor_ckpts)} 个")
    print(f"   reward.ckpt: {len(reward_ckpts)} 个")
    
    issues = []
    
    if len(best_ckpts) == 0:
        print(f"\n   ⚠️  没有best checkpoint")
        print(f"       请检查 rl4seg3d/config/callbacks/model_checkpoint.yaml")
        print(f"       确保 save_top_k >= 1")
        issues.append("缺少best checkpoint")
    else:
        print(f"\n   ✅ Best checkpoints:")
        for ckpt in best_ckpts[:3]:  # 只显示前3个
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"      {ckpt.name} ({size_mb:.1f} MB)")
    
    if len(last_ckpts) == 0:
        print(f"\n   ⚠️  没有last checkpoint")
        issues.append("缺少last checkpoint")
    else:
        print(f"\n   ✅ Last checkpoints:")
        for ckpt in last_ckpts[:3]:
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"      {ckpt.name} ({size_mb:.1f} MB)")
    
    return len(issues) == 0


def diagnose_reward_dataset(log_dir):
    """诊断RewardDataset"""
    print("\n" + "="*70)
    print("3️⃣  RewardDataset诊断")
    print("="*70)
    
    log_dir = Path(log_dir)
    reward_ds = log_dir / "rewardDS"
    
    if not reward_ds.exists():
        print(f"❌ RewardDS目录不存在: {reward_ds}")
        print(f"   可能还没有运行过predict")
        return False
    
    # 检查子目录
    images_dir = reward_ds / "images"
    gt_dir = reward_ds / "gt"
    pred_dir = reward_ds / "pred"
    
    print(f"\n📁 目录结构:")
    
    for dir_name, dir_path in [("images", images_dir), ("gt", gt_dir), ("pred", pred_dir)]:
        if dir_path.exists():
            num_files = len(list(dir_path.glob("*.nii.gz")))
            print(f"   {dir_name:8s}: {num_files} 个文件 ✅")
        else:
            print(f"   {dir_name:8s}: 不存在 ❌")
    
    # 检查样本数量
    if images_dir.exists():
        num_samples = len(list(images_dir.glob("*.nii.gz")))
        print(f"\n📊 样本统计:")
        print(f"   总样本数: {num_samples}")
        
        if num_samples < 10:
            print(f"   ⚠️  样本数太少 (<10)，RewardNet训练可能不充分")
            print(f"       建议: 增加 rl_num_predict 参数")
            return False
        elif num_samples < 50:
            print(f"   ⚠️  样本数较少 (<50)，建议增加更多样本")
            return True
        else:
            print(f"   ✅ 样本数充足 (>= 50)")
            return True
    
    return False


def diagnose_data_quality(log_dir):
    """诊断数据质量"""
    print("\n" + "="*70)
    print("4️⃣  数据质量诊断")
    print("="*70)
    
    log_dir = Path(log_dir)
    reward_ds = log_dir / "rewardDS"
    
    if not reward_ds.exists():
        print(f"❌ RewardDS不存在，跳过数据质量检查")
        return False
    
    try:
        import nibabel as nib
        import numpy as np
        
        images_dir = reward_ds / "images"
        gt_dir = reward_ds / "gt"
        pred_dir = reward_ds / "pred"
        
        if not images_dir.exists():
            print(f"❌ images目录不存在")
            return False
        
        # 随机检查几个文件
        image_files = list(images_dir.glob("*.nii.gz"))
        if len(image_files) == 0:
            print(f"❌ 没有找到图像文件")
            return False
        
        print(f"\n🔍 检查数据形状和范围 (抽样{min(3, len(image_files))}个):")
        
        for img_file in image_files[:3]:
            img = nib.load(img_file).get_fdata()
            gt_file = gt_dir / img_file.name
            pred_file = pred_dir / img_file.name
            
            print(f"\n   文件: {img_file.name}")
            print(f"      图像形状: {img.shape}")
            print(f"      图像范围: [{img.min():.2f}, {img.max():.2f}]")
            
            if gt_file.exists():
                gt = nib.load(gt_file).get_fdata()
                print(f"      GT形状:   {gt.shape}")
                print(f"      GT唯一值: {np.unique(gt)}")
                
                # 检查前景比例
                fg_ratio = (gt > 0).sum() / gt.size * 100
                print(f"      前景比例: {fg_ratio:.2f}%")
                
                if fg_ratio < 1:
                    print(f"      ⚠️  前景比例很小 (<1%)，可能是小目标分割")
                elif fg_ratio > 50:
                    print(f"      ⚠️  前景比例很大 (>50%)，数据可能有问题")
            
            if pred_file.exists():
                pred = nib.load(pred_file).get_fdata()
                pred_fg_ratio = (pred > 0.5).sum() / pred.size * 100
                print(f"      预测前景比例: {pred_fg_ratio:.2f}%")
                
                if gt_file.exists():
                    if pred_fg_ratio > fg_ratio * 10:
                        print(f"      ⚠️  预测前景远大于GT (过度预测)")
        
        print(f"\n   ✅ 数据质量检查完成")
        return True
        
    except ImportError:
        print(f"⚠️  需要nibabel库进行数据质量检查")
        print(f"   安装: pip install nibabel")
        return False
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        return False


def print_recommendations(issues, suggestions):
    """打印改进建议"""
    print("\n" + "="*70)
    print("💡 改进建议")
    print("="*70)
    
    if len(issues) == 0:
        print("\n✅ 一切正常！可以继续训练。")
    else:
        print(f"\n发现 {len(issues)} 个问题:\n")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        if suggestions:
            print(f"\n具体建议:\n")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
    
    print("\n" + "="*70)
    print("🚀 快速优化命令")
    print("="*70)
    print("""
1. 修改配置文件:
   nano rl4seg3d/config/auto_iteration.yaml
   
   建议修改:
   num_iter: 50              # 增加迭代次数
   rl_num_epochs: 50         # 增加训练轮数
   rl_num_predict: 5000      # 增加RewardNet训练数据

2. 重新训练:
   cd /home/ubuntu/RL4Seg3D
   python3 rl4seg3d/auto_iteration.py

3. 评估模型:
   python3 scripts/evaluate_predictions.py

4. 可视化结果:
   python3 scripts/nifti_to_mp4.py \\
       -i /home/ubuntu/my_rl4seg3d_logs/3d_test/rewardDS \\
       -o /home/ubuntu/videos \\
       --reward-dataset --fps 2

5. 搜索最优阈值:
   python3 scripts/find_optimal_threshold.py
""")


def main():
    parser = argparse.ArgumentParser(description='训练状态诊断工具')
    parser.add_argument('--config', type=str,
                       default='rl4seg3d/config/auto_iteration.yaml',
                       help='配置文件路径')
    parser.add_argument('--log-dir', type=str,
                       default='/home/ubuntu/my_rl4seg3d_logs/3d_test',
                       help='日志目录路径')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 训练状态诊断工具")
    print("="*70)
    print(f"\n配置文件: {args.config}")
    print(f"日志目录: {args.log_dir}")
    
    all_issues = []
    all_suggestions = []
    
    # 1. 诊断配置
    config_ok, config_issues, config_suggestions = diagnose_config(Path(args.config))
    all_issues.extend(config_issues)
    all_suggestions.extend(config_suggestions)
    
    # 2. 诊断checkpoint
    ckpt_ok = diagnose_checkpoints(args.log_dir)
    
    # 3. 诊断RewardDataset
    reward_ds_ok = diagnose_reward_dataset(args.log_dir)
    
    # 4. 诊断数据质量
    data_ok = diagnose_data_quality(args.log_dir)
    
    # 5. 打印建议
    print_recommendations(all_issues, all_suggestions)
    
    # 返回状态
    if config_ok and ckpt_ok and reward_ds_ok:
        print("\n✅ 整体状态: 良好")
        sys.exit(0)
    else:
        print("\n⚠️  整体状态: 需要改进")
        sys.exit(1)


if __name__ == "__main__":
    main()

