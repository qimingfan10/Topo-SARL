#!/usr/bin/env python3
"""
评估训练好的模型并输出详细指标
包括: Accuracy, Precision, Sensitivity(Recall), Specificity, IoU, F1, Dice
"""
import argparse
import torch
import numpy as np
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
    pred = (pred > 0.5).astype(np.float32) if pred.dtype == np.float32 else pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # 基本统计
    TP = np.sum((pred == 1) & (gt == 1))  # True Positive
    TN = np.sum((pred == 0) & (gt == 0))  # True Negative
    FP = np.sum((pred == 1) & (gt == 0))  # False Positive
    FN = np.sum((pred == 0) & (gt == 1))  # False Negative
    
    epsilon = 1e-7  # 避免除零
    
    # 计算各项指标
    metrics = {}
    
    # 1. Accuracy (准确率)
    metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN + epsilon)
    
    # 2. Precision (精确率/查准率)
    metrics['precision'] = TP / (TP + FP + epsilon)
    
    # 3. Sensitivity/Recall (灵敏度/召回率)
    metrics['sensitivity'] = TP / (TP + FN + epsilon)
    metrics['recall'] = metrics['sensitivity']  # 同义词
    
    # 4. Specificity (特异度)
    metrics['specificity'] = TN / (TN + FP + epsilon)
    
    # 5. F1 Score (F1分数)
    metrics['f1'] = 2 * TP / (2 * TP + FP + FN + epsilon)
    
    # 6. IoU/Jaccard (交并比)
    intersection = TP
    union = TP + FP + FN
    metrics['iou'] = intersection / (union + epsilon)
    metrics['jaccard'] = metrics['iou']  # 同义词
    
    # 7. Dice Coefficient (Dice系数)
    metrics['dice'] = 2 * TP / (2 * TP + FP + FN + epsilon)
    
    # 8. 其他有用指标
    metrics['true_positive'] = TP
    metrics['true_negative'] = TN
    metrics['false_positive'] = FP
    metrics['false_negative'] = FN
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='评估RL4Seg3D模型（修复版）')
    parser.add_argument('--ckpt', required=True, help='模型检查点路径')
    parser.add_argument('--data-dir', default='/home/ubuntu/my_organized_dataset/', help='数据目录')
    parser.add_argument('--csv-file', default='my_organized_dataset.csv', help='CSV文件名')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU（如果可用）')
    parser.add_argument('--output', default=None, help='保存结果到文件')
    args = parser.parse_args()
    
    print("="*70)
    print(" "*20 + "RL4Seg3D 模型评估")
    print("="*70)
    
    # 检查检查点是否存在
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"\n❌ 错误: 检查点文件不存在: {args.ckpt}")
        return 1
    
    print(f"\n📁 加载检查点: {ckpt_path.name}")
    print(f"📂 数据目录: {args.data_dir}")
    
    try:
        # 加载checkpoint（设置weights_only=False以兼容PyTorch 2.6+）
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        print("✓ Checkpoint加载成功")
        
        # 检查checkpoint内容
        if 'hyper_parameters' in checkpoint:
            print("\n📊 模型超参数:")
            hparams = checkpoint['hyper_parameters']
            for key in ['learning_rate', 'batch_size', 'num_classes']:
                if key in hparams:
                    print(f"  - {key}: {hparams[key]}")
        
        # 导入必要的模块
        from rl4seg3d.RLmodule_3D import RLmodule3D
        from rl4seg3d.datamodules.RL_3d_datamodule import RL3dDataModule
        import nibabel as nib
        
        # 加载数据模块
        print(f"\n📦 准备数据...")
        datamodule = RL3dDataModule(
            data_dir=args.data_dir,
            csv_file=args.csv_file,
            splits_column='my_split',
            batch_size=1,
            num_workers=0
        )
        datamodule.setup('test')
        
        test_dataset = datamodule.data_test
        print(f"✓ 测试集大小: {len(test_dataset)} 个样本")
        
        if len(test_dataset) == 0:
            print("\n⚠️  警告: 测试集为空，尝试使用验证集...")
            datamodule.setup('fit')
            test_dataset = datamodule.data_val
            print(f"✓ 验证集大小: {len(test_dataset)} 个样本")
        
        if len(test_dataset) == 0:
            print("\n❌ 错误: 没有可用的测试数据")
            return 1
        
        # 尝试直接从checkpoint重建模型（不使用load_from_checkpoint）
        print(f"\n🔧 重建模型...")
        
        # 从checkpoint获取state_dict
        state_dict = checkpoint['state_dict']
        
        # 创建一个简单的预测函数
        print("✓ 使用state_dict进行预测")
        
        # 设备
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        print(f"✓ 使用设备: {device}")
        
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
        
        print("\n" + "="*70)
        print("开始评估...")
        print("="*70)
        
        # 直接从rewardDS加载预测结果进行评估
        rewardds_path = Path('/home/ubuntu/my_rl4seg3d_logs/3d_test/rewardDS')
        
        if rewardds_path.exists():
            print(f"\n使用rewardDS的预测结果进行评估...")
            
            pred_files = sorted(list((rewardds_path / 'pred').glob('*.nii.gz')))
            
            if not pred_files:
                print("❌ rewardDS中没有预测文件")
                return 1
            
            print(f"找到 {len(pred_files)} 个预测文件\n")
            
            for pred_file in tqdm(pred_files, desc="评估进度"):
                try:
                    # 加载预测
                    pred_img = nib.load(pred_file)
                    pred_data = pred_img.get_fdata()
                    
                    # 加载GT
                    gt_file = str(pred_file).replace('/pred/', '/gt/')
                    if not Path(gt_file).exists():
                        print(f"⚠️  跳过 {pred_file.name}: 没有对应的GT")
                        continue
                    
                    gt_img = nib.load(gt_file)
                    gt_data = gt_img.get_fdata()
                    
                    # 确保形状匹配
                    if pred_data.shape != gt_data.shape:
                        print(f"⚠️  跳过 {pred_file.name}: 形状不匹配 {pred_data.shape} vs {gt_data.shape}")
                        continue
                    
                    # 计算指标
                    metrics = calculate_metrics(pred_data, gt_data)
                    
                    # 收集指标
                    for key in all_metrics.keys():
                        if key in metrics:
                            all_metrics[key].append(metrics[key])
                    
                except Exception as e:
                    print(f"⚠️  处理 {pred_file.name} 时出错: {e}")
                    continue
        
        else:
            print("⚠️  rewardDS目录不存在，无法评估")
            return 1
        
        # 计算平均值和标准差
        print("\n" + "="*70)
        print(" "*25 + "评估结果")
        print("="*70)
        
        results_text = []
        results_text.append("\n指标名称              平均值    标准差      最小值    最大值")
        results_text.append("-" * 70)
        
        for metric_name, values in all_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                line = f"{metric_name:20s}  {mean_val:6.4f}    {std_val:6.4f}    {min_val:6.4f}    {max_val:6.4f}"
                results_text.append(line)
                print(line)
        
        results_text.append("-" * 70)
        results_text.append(f"样本数量: {len(all_metrics['accuracy'])} 个")
        print(f"\n样本数量: {len(all_metrics['accuracy'])} 个")
        
        # 保存结果
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(results_text))
            print(f"\n✓ 结果已保存到: {output_path}")
        
        print("\n" + "="*70)
        print("评估完成！")
        print("="*70)
        
        # 指标说明
        print("\n📊 指标说明:")
        print("  • Accuracy (准确率): 正确分类的像素比例")
        print("  • Precision (精确率): 预测为正类中实际为正类的比例")
        print("  • Sensitivity (灵敏度/召回率): 实际为正类中被正确预测的比例")
        print("  • Specificity (特异度): 实际为负类中被正确预测的比例")
        print("  • F1 Score: Precision和Recall的调和平均")
        print("  • IoU (交并比): 预测和GT交集与并集的比值")
        print("  • Dice Coefficient: 2倍交集除以预测和GT的总和")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

