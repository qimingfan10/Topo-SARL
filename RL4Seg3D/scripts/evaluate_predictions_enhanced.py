#!/usr/bin/env python3
"""
增强版评估脚本
添加更多详细指标用于深入分析模型问题
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json

def calculate_metrics_enhanced(pred, gt):
    """
    计算增强的分割指标
    包括基础指标、统计分析、边界指标等
    """
    # 确保是二值数据
    pred = (pred > 0.5).astype(np.uint8)
    gt = gt.astype(np.uint8)
    
    # 基本统计
    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    
    total_pixels = pred.size
    epsilon = 1e-7
    
    metrics = {}
    
    # === 基础指标 ===
    metrics['accuracy'] = (TP + TN) / (total_pixels + epsilon)
    metrics['precision'] = TP / (TP + FP + epsilon)
    metrics['sensitivity'] = TP / (TP + FN + epsilon)
    metrics['specificity'] = TN / (TN + FP + epsilon)
    metrics['f1'] = 2 * TP / (2 * TP + FP + FN + epsilon)
    metrics['iou'] = TP / (TP + FP + FN + epsilon)
    metrics['dice'] = 2 * TP / (2 * TP + FP + FN + epsilon)
    
    # === 统计指标 ===
    metrics['TP'] = int(TP)
    metrics['TN'] = int(TN)
    metrics['FP'] = int(FP)
    metrics['FN'] = int(FN)
    metrics['total_pixels'] = int(total_pixels)
    
    # 前景/背景比例
    gt_fg_pixels = int(np.sum(gt))
    pred_fg_pixels = int(np.sum(pred))
    
    metrics['gt_foreground_ratio'] = gt_fg_pixels / total_pixels
    metrics['pred_foreground_ratio'] = pred_fg_pixels / total_pixels
    metrics['gt_foreground_pixels'] = gt_fg_pixels
    metrics['pred_foreground_pixels'] = pred_fg_pixels
    
    # === 错误分析 ===
    # False Positive Rate (假阳性率)
    metrics['fpr'] = FP / (FP + TN + epsilon)
    
    # False Negative Rate (假阴性率/漏检率)
    metrics['fnr'] = FN / (FN + TP + epsilon)
    
    # 过度预测比例 (预测比GT多多少)
    metrics['over_prediction_ratio'] = (pred_fg_pixels - gt_fg_pixels) / (gt_fg_pixels + epsilon)
    
    # 预测覆盖率 (预测了多少真实前景)
    metrics['coverage'] = TP / (gt_fg_pixels + epsilon)
    
    # 预测纯度 (预测中有多少是正确的)
    metrics['purity'] = TP / (pred_fg_pixels + epsilon) if pred_fg_pixels > 0 else 0.0
    
    # === 像素级统计 ===
    # 计算预测分布的偏差
    if gt_fg_pixels > 0:
        # 过度预测倍数
        metrics['fg_prediction_fold'] = pred_fg_pixels / gt_fg_pixels
    else:
        metrics['fg_prediction_fold'] = float('inf') if pred_fg_pixels > 0 else 1.0
    
    # === 分类质量 ===
    # Matthews Correlation Coefficient (MCC)
    mcc_num = (TP * TN - FP * FN)
    mcc_den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    metrics['mcc'] = mcc_num / (mcc_den + epsilon)
    
    # Balanced Accuracy
    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
    
    # === 体积相关 ===
    metrics['volume_similarity'] = 1 - abs(pred_fg_pixels - gt_fg_pixels) / (pred_fg_pixels + gt_fg_pixels + epsilon)
    
    return metrics


def analyze_per_slice(pred, gt):
    """逐slice分析（如果是3D数据）"""
    if len(pred.shape) == 3:
        slice_metrics = []
        for i in range(pred.shape[2]):
            pred_slice = pred[:, :, i]
            gt_slice = gt[:, :, i]
            
            # 跳过空slice
            if gt_slice.sum() == 0 and pred_slice.sum() == 0:
                continue
            
            metrics = calculate_metrics_enhanced(pred_slice, gt_slice)
            metrics['slice_idx'] = i
            slice_metrics.append(metrics)
        
        return slice_metrics
    return []


def main():
    parser = argparse.ArgumentParser(description='增强版评估工具')
    parser.add_argument('--reward-ds-path', type=str,
                       default='/home/ubuntu/my_rl4seg3d_logs/3d_test/rewardDS',
                       help='RewardDS目录路径')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ubuntu/evaluation_enhanced',
                       help='结果保存目录')
    parser.add_argument('--analyze-slices', action='store_true',
                       help='是否进行逐slice分析')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" "*25 + "增强版分割评估工具")
    print("="*80)
    
    reward_ds_path = Path(args.reward_ds_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_dir = reward_ds_path / 'pred'
    gt_dir = reward_ds_path / 'gt'
    images_dir = reward_ds_path / 'images'
    
    if not pred_dir.exists() or not gt_dir.exists():
        print(f"\n❌ 错误: pred或gt目录不存在")
        return 1
    
    pred_files = sorted(list(pred_dir.glob('*.nii.gz')))
    
    if not pred_files:
        print(f"\n❌ 错误: 没有找到预测文件")
        return 1
    
    print(f"\n📂 数据路径: {reward_ds_path}")
    print(f"📊 找到 {len(pred_files)} 个样本")
    print(f"💾 结果保存到: {output_dir}")
    
    # 收集所有指标
    all_metrics = []
    all_slice_metrics = []
    
    print("\n" + "="*80)
    print("开始详细评估...")
    print("="*80 + "\n")
    
    for pred_file in tqdm(pred_files, desc="评估进度"):
        try:
            # 加载数据
            pred = nib.load(pred_file).get_fdata()
            gt_file = gt_dir / pred_file.name
            
            if not gt_file.exists():
                continue
            
            gt = nib.load(gt_file).get_fdata()
            
            if pred.shape != gt.shape:
                print(f"⚠️  形状不匹配: {pred_file.name}")
                continue
            
            # 计算整体指标
            metrics = calculate_metrics_enhanced(pred, gt)
            metrics['filename'] = pred_file.name
            metrics['shape'] = pred.shape
            
            all_metrics.append(metrics)
            
            # 逐slice分析
            if args.analyze_slices:
                slice_metrics = analyze_per_slice(pred, gt)
                for sm in slice_metrics:
                    sm['filename'] = pred_file.name
                all_slice_metrics.extend(slice_metrics)
        
        except Exception as e:
            print(f"❌ 错误 {pred_file.name}: {e}")
            continue
    
    if not all_metrics:
        print("\n❌ 没有成功评估任何样本")
        return 1
    
    # === 生成详细报告 ===
    print("\n" + "="*80)
    print(" "*25 + "详细评估结果")
    print("="*80)
    
    # 1. 基础指标统计
    print("\n📊 基础分割指标:")
    print("-"*80)
    metric_names = ['dice', 'iou', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1']
    print(f"{'指标':<15} {'平均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    print("-"*80)
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        print(f"{metric.upper():<15} {np.mean(values):10.4f} {np.std(values):10.4f} "
              f"{np.min(values):10.4f} {np.max(values):10.4f}")
    
    # 2. 错误分析
    print("\n❌ 错误分析:")
    print("-"*80)
    fpr_values = [m['fpr'] for m in all_metrics]
    fnr_values = [m['fnr'] for m in all_metrics]
    print(f"假阳性率 (FPR):        {np.mean(fpr_values):.4f} ± {np.std(fpr_values):.4f}")
    print(f"假阴性率 (FNR):        {np.mean(fnr_values):.4f} ± {np.std(fnr_values):.4f}")
    
    # 3. 前景比例分析
    print("\n🎯 前景比例分析:")
    print("-"*80)
    gt_fg_ratios = [m['gt_foreground_ratio'] for m in all_metrics]
    pred_fg_ratios = [m['pred_foreground_ratio'] for m in all_metrics]
    over_pred_ratios = [m['over_prediction_ratio'] for m in all_metrics]
    fg_folds = [m['fg_prediction_fold'] for m in all_metrics if m['fg_prediction_fold'] != float('inf')]
    
    print(f"GT前景比例:            {np.mean(gt_fg_ratios)*100:.2f}% ± {np.std(gt_fg_ratios)*100:.2f}%")
    print(f"预测前景比例:          {np.mean(pred_fg_ratios)*100:.2f}% ± {np.std(pred_fg_ratios)*100:.2f}%")
    print(f"过度预测比例:          {np.mean(over_pred_ratios)*100:.1f}% (相对GT)")
    print(f"预测倍数:              {np.mean(fg_folds):.2f}x (预测是GT的几倍)")
    
    # 4. 预测质量分析
    print("\n✅ 预测质量分析:")
    print("-"*80)
    coverages = [m['coverage'] for m in all_metrics]
    purities = [m['purity'] for m in all_metrics]
    print(f"覆盖率 (捕获了多少GT):  {np.mean(coverages)*100:.2f}%")
    print(f"纯度 (预测中正确比例):  {np.mean(purities)*100:.2f}%")
    
    # 5. 混淆矩阵统计
    print("\n📈 混淆矩阵统计 (总和):")
    print("-"*80)
    total_TP = sum(m['TP'] for m in all_metrics)
    total_FP = sum(m['FP'] for m in all_metrics)
    total_FN = sum(m['FN'] for m in all_metrics)
    total_TN = sum(m['TN'] for m in all_metrics)
    total_all = total_TP + total_FP + total_FN + total_TN
    
    print(f"True Positive (TP):    {total_TP:,} ({total_TP/total_all*100:.2f}%)")
    print(f"False Positive (FP):   {total_FP:,} ({total_FP/total_all*100:.2f}%)")
    print(f"False Negative (FN):   {total_FN:,} ({total_FN/total_all*100:.2f}%)")
    print(f"True Negative (TN):    {total_TN:,} ({total_TN/total_all*100:.2f}%)")
    
    # 6. 每个样本的详细指标
    print("\n📋 每个样本的详细指标:")
    print("-"*80)
    print(f"{'文件名':<35} {'Dice':>7} {'前景比(GT)':>12} {'前景比(Pred)':>14} {'过度预测':>10}")
    print("-"*80)
    for m in all_metrics:
        print(f"{m['filename']:<35} {m['dice']:7.4f} "
              f"{m['gt_foreground_ratio']*100:10.2f}% "
              f"{m['pred_foreground_ratio']*100:12.2f}% "
              f"{m['over_prediction_ratio']*100:9.1f}%")
    
    # === 保存详细结果 ===
    # 保存JSON
    json_path = output_dir / 'evaluation_detailed.json'
    with open(json_path, 'w') as f:
        json.dump({
            'summary': {
                'num_samples': len(all_metrics),
                'mean_dice': float(np.mean([m['dice'] for m in all_metrics])),
                'mean_iou': float(np.mean([m['iou'] for m in all_metrics])),
                'mean_precision': float(np.mean([m['precision'] for m in all_metrics])),
                'mean_sensitivity': float(np.mean([m['sensitivity'] for m in all_metrics])),
                'mean_gt_fg_ratio': float(np.mean(gt_fg_ratios)),
                'mean_pred_fg_ratio': float(np.mean(pred_fg_ratios)),
                'mean_over_prediction': float(np.mean(over_pred_ratios)),
            },
            'per_sample_metrics': all_metrics
        }, f, indent=2, default=str)
    
    # 保存文本报告
    txt_path = output_dir / 'evaluation_report.txt'
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" "*25 + "增强版评估报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"样本数: {len(all_metrics)}\n")
        f.write(f"数据路径: {reward_ds_path}\n\n")
        
        f.write("基础指标:\n")
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            f.write(f"  {metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        
        f.write(f"\n前景比例:\n")
        f.write(f"  GT前景: {np.mean(gt_fg_ratios)*100:.2f}%\n")
        f.write(f"  预测前景: {np.mean(pred_fg_ratios)*100:.2f}%\n")
        f.write(f"  过度预测: {np.mean(over_pred_ratios)*100:.1f}%\n")
        
        f.write(f"\n混淆矩阵:\n")
        f.write(f"  TP: {total_TP:,}, FP: {total_FP:,}\n")
        f.write(f"  FN: {total_FN:,}, TN: {total_TN:,}\n")
    
    print(f"\n✅ 详细结果已保存:")
    print(f"   JSON: {json_path}")
    print(f"   文本: {txt_path}")
    
    # === 诊断建议 ===
    print("\n" + "="*80)
    print(" "*25 + "🔍 诊断和建议")
    print("="*80)
    
    avg_dice = np.mean([m['dice'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_sensitivity = np.mean([m['sensitivity'] for m in all_metrics])
    avg_over_pred = np.mean(over_pred_ratios)
    
    issues = []
    
    if avg_dice < 0.3:
        issues.append(("🚨 严重", f"Dice Score过低 ({avg_dice:.1%})"))
    elif avg_dice < 0.6:
        issues.append(("⚠️  中等", f"Dice Score偏低 ({avg_dice:.1%})"))
    
    if avg_precision < 0.1:
        issues.append(("🚨 严重", f"Precision极低 ({avg_precision:.1%}) - 大量假阳性"))
    elif avg_precision < 0.5:
        issues.append(("⚠️  中等", f"Precision偏低 ({avg_precision:.1%}) - 较多假阳性"))
    
    if avg_over_pred > 10:
        issues.append(("🚨 严重", f"严重过度预测 (预测是GT的{avg_over_pred:.0f}倍)"))
    elif avg_over_pred > 3:
        issues.append(("⚠️  中等", f"明显过度预测 (预测是GT的{avg_over_pred:.1f}倍)"))
    
    if avg_sensitivity < 0.5:
        issues.append(("⚠️  注意", f"Sensitivity偏低 ({avg_sensitivity:.1%}) - 漏检较多"))
    
    if issues:
        print("\n发现的问题:")
        for severity, issue in issues:
            print(f"  {severity}: {issue}")
    else:
        print("\n✅ 模型表现良好！")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
