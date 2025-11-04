#!/usr/bin/env python3
"""
直接评估预测结果（从rewardDS目录）
计算所有分割指标：Accuracy, Precision, Sensitivity, Specificity, IoU, F1, Dice
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def calculate_metrics(pred, gt):
    """
    计算完整的分割指标
    
    参数:
        pred: 预测结果 (numpy array, 二值0/1)
        gt: Ground Truth (numpy array, 二值0/1)
    
    返回:
        dict: 包含所有指标的字典
    """
    # 确保是二值数据
    pred = (pred > 0.5).astype(np.float32) if pred.max() > 1 else pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # 基本统计
    TP = np.sum((pred == 1) & (gt == 1))  # True Positive
    TN = np.sum((pred == 0) & (gt == 0))  # True Negative  
    FP = np.sum((pred == 1) & (gt == 0))  # False Positive
    FN = np.sum((pred == 0) & (gt == 1))  # False Negative
    
    epsilon = 1e-7  # 避免除零
    
    # 计算各项指标
    metrics = {}
    
    # 1. Accuracy (ACC - 准确率)
    metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN + epsilon)
    
    # 2. Precision (PRE - 精确率/查准率)
    metrics['precision'] = TP / (TP + FP + epsilon)
    
    # 3. Sensitivity/Recall (SEN - 灵敏度/召回率)
    metrics['sensitivity'] = TP / (TP + FN + epsilon)
    metrics['recall'] = metrics['sensitivity']  # 同义词
    
    # 4. Specificity (SPE - 特异度)
    metrics['specificity'] = TN / (TN + FP + epsilon)
    
    # 5. F1 Score (F1分数)
    metrics['f1'] = 2 * TP / (2 * TP + FP + FN + epsilon)
    
    # 6. IoU/Jaccard (IOU - 交并比)
    intersection = TP
    union = TP + FP + FN
    metrics['iou'] = intersection / (union + epsilon)
    metrics['jaccard'] = metrics['iou']  # 同义词
    
    # 7. Dice Coefficient (DSC - Dice系数)
    metrics['dice'] = 2 * TP / (2 * TP + FP + FN + epsilon)
    
    # 基本统计
    metrics['TP'] = TP
    metrics['TN'] = TN
    metrics['FP'] = FP
    metrics['FN'] = FN
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='评估预测结果（rewardDS）')
    parser.add_argument('--rewardds-dir', default='/home/ubuntu/my_rl4seg3d_logs/3d_test/rewardDS',
                       help='rewardDS目录路径')
    parser.add_argument('--output', default='/home/ubuntu/evaluation_results.txt',
                       help='保存结果到文件')
    parser.add_argument('--verbose', action='store_true', help='显示每个样本的详细信息')
    args = parser.parse_args()
    
    print("="*80)
    print(" "*25 + "分割模型评估工具")
    print("="*80)
    
    rewardds_path = Path(args.rewardds_dir)
    
    if not rewardds_path.exists():
        print(f"\n❌ 错误: rewardDS目录不存在: {rewardds_path}")
        return 1
    
    pred_dir = rewardds_path / 'pred'
    gt_dir = rewardds_path / 'gt'
    
    if not pred_dir.exists():
        print(f"\n❌ 错误: 预测目录不存在: {pred_dir}")
        return 1
    
    if not gt_dir.exists():
        print(f"\n❌ 错误: GT目录不存在: {gt_dir}")
        return 1
    
    # 查找所有预测文件
    pred_files = sorted(list(pred_dir.glob('*.nii.gz')))
    
    if not pred_files:
        print(f"\n❌ 错误: 预测目录为空: {pred_dir}")
        return 1
    
    print(f"\n📂 rewardDS目录: {rewardds_path}")
    print(f"📊 找到 {len(pred_files)} 个预测文件")
    
    # 收集所有指标
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'sensitivity': [],
        'specificity': [],
        'f1': [],
        'iou': [],
        'dice': []
    }
    
    sample_details = []
    
    print("\n" + "="*80)
    print("开始评估...")
    print("="*80 + "\n")
    
    # 遍历所有预测文件
    for pred_file in tqdm(pred_files, desc="评估进度", ncols=80):
        try:
            # 加载预测
            pred_img = nib.load(pred_file)
            pred_data = pred_img.get_fdata()
            
            # 构造GT文件路径
            gt_file = gt_dir / pred_file.name
            
            if not gt_file.exists():
                if args.verbose:
                    print(f"⚠️  跳过 {pred_file.name}: 没有对应的GT")
                continue
            
            # 加载GT
            gt_img = nib.load(gt_file)
            gt_data = gt_img.get_fdata()
            
            # 确保形状匹配
            if pred_data.shape != gt_data.shape:
                if args.verbose:
                    print(f"⚠️  跳过 {pred_file.name}: 形状不匹配 {pred_data.shape} vs {gt_data.shape}")
                continue
            
            # 计算指标
            metrics = calculate_metrics(pred_data, gt_data)
            
            # 收集指标
            for key in all_metrics.keys():
                if key in metrics:
                    all_metrics[key].append(metrics[key])
            
            # 保存详细信息
            sample_details.append({
                'filename': pred_file.name,
                'shape': pred_data.shape,
                **metrics
            })
            
            if args.verbose:
                print(f"✓ {pred_file.name}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, F1={metrics['f1']:.4f}")
        
        except Exception as e:
            print(f"❌ 处理 {pred_file.name} 时出错: {e}")
            continue
    
    # 计算统计结果
    if not all_metrics['accuracy']:
        print("\n❌ 错误: 没有成功评估任何样本")
        return 1
    
    print("\n" + "="*80)
    print(" "*30 + "评估结果汇总")
    print("="*80)
    
    results_lines = []
    results_lines.append("\n" + "="*80)
    results_lines.append(" "*30 + "评估结果汇总")
    results_lines.append("="*80)
    results_lines.append("")
    results_lines.append("指标缩写说明:")
    results_lines.append("  ACC  - Accuracy      (准确率)")
    results_lines.append("  PRE  - Precision     (精确率)")
    results_lines.append("  SEN  - Sensitivity   (灵敏度/召回率)")
    results_lines.append("  SPE  - Specificity   (特异度)")
    results_lines.append("  F1   - F1 Score      (F1分数)")
    results_lines.append("  IOU  - IoU/Jaccard   (交并比)")
    results_lines.append("  DSC  - Dice          (Dice系数)")
    results_lines.append("")
    results_lines.append("="*80)
    results_lines.append("")
    
    # 表头
    header = f"{'指标':<15} {'平均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}"
    results_lines.append(header)
    results_lines.append("-"*80)
    print(header)
    print("-"*80)
    
    # 定义指标显示顺序和名称
    metrics_display = [
        ('accuracy', 'ACC'),
        ('precision', 'PRE'),
        ('sensitivity', 'SEN'),
        ('specificity', 'SPE'),
        ('f1', 'F1'),
        ('iou', 'IOU'),
        ('dice', 'DSC')
    ]
    
    for metric_key, metric_name in metrics_display:
        if metric_key in all_metrics and all_metrics[metric_key]:
            values = all_metrics[metric_key]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            line = f"{metric_name:<15} {mean_val:10.4f} {std_val:10.4f} {min_val:10.4f} {max_val:10.4f}"
            results_lines.append(line)
            print(line)
    
    results_lines.append("-"*80)
    results_lines.append(f"评估样本数: {len(all_metrics['accuracy'])} 个")
    results_lines.append("="*80)
    
    print("-"*80)
    print(f"评估样本数: {len(all_metrics['accuracy'])} 个")
    print("="*80)
    
    # 添加每个样本的详细信息
    if sample_details:
        results_lines.append("")
        results_lines.append("")
        results_lines.append("="*80)
        results_lines.append(" "*25 + "每个样本的详细指标")
        results_lines.append("="*80)
        results_lines.append("")
        
        detail_header = f"{'文件名':<40} {'Dice':>8} {'IoU':>8} {'F1':>8} {'Acc':>8}"
        results_lines.append(detail_header)
        results_lines.append("-"*80)
        
        for detail in sample_details:
            detail_line = f"{detail['filename']:<40} {detail['dice']:8.4f} {detail['iou']:8.4f} {detail['f1']:8.4f} {detail['accuracy']:8.4f}"
            results_lines.append(detail_line)
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_lines))
    
    print(f"\n✓ 完整结果已保存到: {output_path}")
    
    # 简单的性能评价
    print("\n" + "="*80)
    print(" "*30 + "性能评价")
    print("="*80)
    
    avg_dice = np.mean(all_metrics['dice'])
    avg_iou = np.mean(all_metrics['iou'])
    
    if avg_dice >= 0.8:
        quality = "优秀 ⭐⭐⭐"
    elif avg_dice >= 0.7:
        quality = "良好 ⭐⭐"
    elif avg_dice >= 0.5:
        quality = "中等 ⭐"
    else:
        quality = "需要改进"
    
    print(f"\n  整体Dice Score: {avg_dice:.4f}")
    print(f"  整体IoU Score:  {avg_iou:.4f}")
    print(f"  模型质量评级:   {quality}")
    print("\n" + "="*80)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

