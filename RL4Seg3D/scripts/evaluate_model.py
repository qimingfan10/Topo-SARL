#!/usr/bin/env python3
"""评估训练好的模型并输出详细指标"""
import argparse
import torch
import numpy as np
from pytorch_lightning import Trainer
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='评估RL4Seg3D模型')
    parser.add_argument('--ckpt', required=True, help='模型检查点路径')
    parser.add_argument('--data-dir', default='/home/ubuntu/my_organized_dataset/', help='数据目录')
    parser.add_argument('--csv-file', default='my_organized_dataset.csv', help='CSV文件名')
    parser.add_argument('--devices', type=int, default=1, help='使用的GPU数量')
    args = parser.parse_args()
    
    print("="*60)
    print("RL4Seg3D 模型评估")
    print("="*60)
    
    # 检查检查点是否存在
    if not Path(args.ckpt).exists():
        print(f"错误: 检查点文件不存在: {args.ckpt}")
        return 1
    
    print(f"\n加载模型: {args.ckpt}")
    
    try:
        # 动态导入以避免初始化问题
        from rl4seg3d.RLmodule_3D import RLmodule3D
        from rl4seg3d.datamodules.RL_3d_datamodule import RL3dDataModule
        
        # 加载模型
        model = RLmodule3D.load_from_checkpoint(args.ckpt, map_location='cpu')
        model.eval()
        print("✓ 模型加载成功")
        
        # 加载数据
        print(f"\n加载数据: {args.data_dir}")
        datamodule = RL3dDataModule(
            data_dir=args.data_dir,
            csv_file=args.csv_file,
            splits_column='my_split',
            batch_size=1,
            num_workers=0
        )
        
        # Setup数据
        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()
        print(f"✓ 数据加载成功 (测试集大小: {len(test_loader.dataset)})")
        
        # 手动评估（避免Lightning的设备问题）
        print("\n开始评估...")
        print("-"*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() and args.devices > 0 else 'cpu')
        model = model.to(device)
        
        all_dice = []
        all_acc = []
        all_rewards = []
        
        for batch_idx, batch in enumerate(test_loader):
            b_img = batch.get('img')
            b_gt = batch.get('gt')
            
            if b_img is None:
                continue
            
            # 移到设备
            b_img = b_img.to(device)
            if b_gt is not None:
                b_gt = b_gt.to(device)
            
            # 预测
            with torch.no_grad():
                # 处理维度
                if b_img.dim() == 6 and b_img.shape[0] == 1:
                    b_img = b_img.squeeze(0)
                if b_gt is not None and b_gt.dim() == 5 and b_gt.shape[0] == 1:
                    b_gt = b_gt.squeeze(0)
                
                # 获取预测
                y_pred = model.actor.act(b_img, sample=False)
                
                if b_gt is not None:
                    # 计算Dice
                    y_pred_binary = (y_pred > 0.5).long() if y_pred.dtype == torch.float else y_pred
                    b_gt_binary = b_gt.long()
                    
                    intersection = ((y_pred_binary == 1) & (b_gt_binary == 1)).sum().float()
                    dice = (2. * intersection) / (y_pred_binary.sum() + b_gt_binary.sum() + 1e-8)
                    
                    # 计算准确率
                    acc = (y_pred_binary == b_gt_binary).float().mean()
                    
                    all_dice.append(dice.item())
                    all_acc.append(acc.item())
                    
                    # 尝试计算reward
                    try:
                        rewards = model.reward_func(y_pred, b_img, b_gt)
                        if isinstance(rewards, (list, tuple)):
                            reward_val = rewards[0].mean().item()
                        else:
                            reward_val = rewards.mean().item()
                        all_rewards.append(reward_val)
                    except:
                        pass
                    
                    print(f"样本 {batch_idx+1}: Dice={dice.item():.4f}, Acc={acc.item():.4f}")
        
        # 输出结果
        print("\n" + "="*60)
        print("评估结果汇总")
        print("="*60)
        
        if all_dice:
            print(f"Dice Score    : {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
            print(f"Accuracy      : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
            if all_rewards:
                print(f"Reward        : {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
            print(f"测试样本数    : {len(all_dice)}")
        else:
            print("警告: 没有Ground Truth，无法计算指标")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

