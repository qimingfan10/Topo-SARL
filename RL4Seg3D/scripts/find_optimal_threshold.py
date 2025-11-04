#!/usr/bin/env python3
"""
搜索最优分割阈值
自动测试不同阈值，找到Dice最高的阈值
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_metrics(pred, gt):
    """计算各项指标"""
    pred = pred.flatten()
    gt = gt.flatten()
    
    tp = ((pred == 1) & (gt == 1)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    
    # 避免除零
    epsilon = 1e-8
    
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'sensitivity': sensitivity
    }


def find_optimal_threshold(reward_ds_path, output_dir=None):
    """
    搜索最优阈值
    
    Args:
        reward_ds_path: rewardDS目录路径
        output_dir: 结果保存目录
    """
    reward_ds_path = Path(reward_ds_path)
    pred_dir = reward_ds_path / "pred"
    gt_dir = reward_ds_path / "gt"
    
    if not pred_dir.exists() or not gt_dir.exists():
        print(f"错误: 找不到pred或gt目录")
        print(f"  pred: {pred_dir}")
        print(f"  gt: {gt_dir}")
        return
    
    # 获取所有文件
    pred_files = sorted(list(pred_dir.glob("*.nii.gz")))
    if len(pred_files) == 0:
        print(f"错误: {pred_dir} 中没有.nii.gz文件")
        return
    
    print(f"找到 {len(pred_files)} 个预测文件")
    print("="*70)
    
    # 测试不同阈值
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = {
        'threshold': [],
        'dice': [],
        'iou': [],
        'precision': [],
        'sensitivity': []
    }
    
    print("\n开始测试不同阈值...")
    print(f"{'阈值':<10} {'Dice':<10} {'IoU':<10} {'Precision':<12} {'Sensitivity':<12}")
    print("-"*70)
    
    for threshold in tqdm(thresholds, desc="测试阈值"):
        metrics_list = []
        
        for pred_file in pred_files:
            gt_file = gt_dir / pred_file.name
            
            if not gt_file.exists():
                continue
            
            # 加载数据
            pred = nib.load(pred_file).get_fdata()
            gt = nib.load(gt_file).get_fdata()
            
            # 应用阈值
            pred_binary = (pred > threshold).astype(np.uint8)
            gt_binary = (gt > 0.5).astype(np.uint8)
            
            # 计算指标
            metrics = compute_metrics(pred_binary, gt_binary)
            metrics_list.append(metrics)
        
        # 计算平均值
        if len(metrics_list) > 0:
            mean_metrics = {
                k: np.mean([m[k] for m in metrics_list])
                for k in metrics_list[0].keys()
            }
            
            results['threshold'].append(threshold)
            results['dice'].append(mean_metrics['dice'])
            results['iou'].append(mean_metrics['iou'])
            results['precision'].append(mean_metrics['precision'])
            results['sensitivity'].append(mean_metrics['sensitivity'])
            
            print(f"{threshold:<10.2f} "
                  f"{mean_metrics['dice']:<10.4f} "
                  f"{mean_metrics['iou']:<10.4f} "
                  f"{mean_metrics['precision']:<12.4f} "
                  f"{mean_metrics['sensitivity']:<12.4f}")
    
    # 找到最优阈值
    best_idx = np.argmax(results['dice'])
    best_threshold = results['threshold'][best_idx]
    best_dice = results['dice'][best_idx]
    best_iou = results['iou'][best_idx]
    
    print("\n" + "="*70)
    print(f"🎯 最优阈值: {best_threshold:.2f}")
    print(f"   Dice Score: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print(f"   IoU Score:  {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"   Precision:  {results['precision'][best_idx]:.4f}")
    print(f"   Sensitivity: {results['sensitivity'][best_idx]:.4f}")
    print("="*70)
    
    # 绘制曲线
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(results['threshold'], results['dice'], 'b-', linewidth=2)
        plt.axvline(best_threshold, color='r', linestyle='--', label=f'最优阈值={best_threshold:.2f}')
        plt.xlabel('阈值')
        plt.ylabel('Dice Score')
        plt.title('Dice Score vs 阈值')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(results['threshold'], results['iou'], 'g-', linewidth=2)
        plt.axvline(best_threshold, color='r', linestyle='--')
        plt.xlabel('阈值')
        plt.ylabel('IoU Score')
        plt.title('IoU Score vs 阈值')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(results['threshold'], results['precision'], 'orange', linewidth=2, label='Precision')
        plt.plot(results['threshold'], results['sensitivity'], 'purple', linewidth=2, label='Sensitivity')
        plt.axvline(best_threshold, color='r', linestyle='--')
        plt.xlabel('阈值')
        plt.ylabel('分数')
        plt.title('Precision & Sensitivity vs 阈值')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(results['precision'], results['sensitivity'], 'b-', linewidth=2)
        plt.scatter([results['precision'][best_idx]], [results['sensitivity'][best_idx]], 
                   color='r', s=100, zorder=5, label=f'最优点 (阈值={best_threshold:.2f})')
        plt.xlabel('Precision')
        plt.ylabel('Sensitivity (Recall)')
        plt.title('Precision-Recall 曲线')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        plot_path = output_dir / 'threshold_optimization.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 曲线图已保存到: {plot_path}")
        
        # 保存结果到文本文件
        result_file = output_dir / 'optimal_threshold.txt'
        with open(result_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("最优阈值搜索结果\n")
            f.write("="*70 + "\n\n")
            f.write(f"数据集: {reward_ds_path}\n")
            f.write(f"样本数: {len(pred_files)}\n")
            f.write(f"测试阈值范围: {thresholds[0]:.2f} - {thresholds[-1]:.2f}\n\n")
            f.write(f"🎯 最优阈值: {best_threshold:.2f}\n\n")
            f.write(f"性能指标:\n")
            f.write(f"  Dice Score:  {best_dice:.4f} ({best_dice*100:.2f}%)\n")
            f.write(f"  IoU Score:   {best_iou:.4f} ({best_iou*100:.2f}%)\n")
            f.write(f"  Precision:   {results['precision'][best_idx]:.4f}\n")
            f.write(f"  Sensitivity: {results['sensitivity'][best_idx]:.4f}\n\n")
            f.write("="*70 + "\n\n")
            f.write("详细结果:\n\n")
            f.write(f"{'阈值':<10} {'Dice':<10} {'IoU':<10} {'Precision':<12} {'Sensitivity':<12}\n")
            f.write("-"*70 + "\n")
            for i in range(len(results['threshold'])):
                f.write(f"{results['threshold'][i]:<10.2f} "
                       f"{results['dice'][i]:<10.4f} "
                       f"{results['iou'][i]:<10.4f} "
                       f"{results['precision'][i]:<12.4f} "
                       f"{results['sensitivity'][i]:<12.4f}\n")
        
        print(f"📄 结果已保存到: {result_file}")
    
    return best_threshold, best_dice


def main():
    parser = argparse.ArgumentParser(description='搜索最优分割阈值')
    parser.add_argument('--reward-ds-path', type=str,
                       default='/home/ubuntu/my_rl4seg3d_logs/3d_test/rewardDS',
                       help='RewardDS目录路径')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ubuntu/threshold_optimization',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("最优阈值搜索工具")
    print("="*70)
    print(f"\n数据路径: {args.reward_ds_path}")
    print(f"输出目录: {args.output_dir}\n")
    
    best_threshold, best_dice = find_optimal_threshold(
        args.reward_ds_path,
        args.output_dir
    )
    
    print(f"\n💡 如何使用最优阈值:")
    print(f"   在 RLmodule_3D.py 的 predict_step 中:")
    print(f"   actions = (logits > {best_threshold:.2f}).float()")
    print()


if __name__ == "__main__":
    main()

