"""
推理和可视化脚本
"""
import argparse
import os
import sys
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, '/home/ubuntu/sam+RL')

from models.sam2_wrapper import SAM2CandidateGenerator
from rewards.reward_functions import RewardCalculator, compute_iou, compute_dice, compute_cldice
from env.candidate_selection_env import CandidateSelectionEnv
from utils.data_loader import VesselDataset
from stable_baselines3 import PPO


def visualize_result(image, gt_mask, pred_mask, metrics, save_path):
    """
    可视化分割结果
    
    Args:
        image: 原始图像 (H, W, 3)
        gt_mask: 真实标注 (H, W)
        pred_mask: 预测结果 (H, W)
        metrics: 评估指标字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 真实标注
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')
    
    # 预测结果
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title('Prediction', fontsize=14)
    axes[0, 2].axis('off')
    
    # 叠加显示 - GT
    overlay_gt = image.copy()
    overlay_gt[gt_mask] = overlay_gt[gt_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    axes[1, 0].imshow(overlay_gt.astype(np.uint8))
    axes[1, 0].set_title('GT Overlay (Green)', fontsize=14)
    axes[1, 0].axis('off')
    
    # 叠加显示 - Pred
    overlay_pred = image.copy()
    overlay_pred[pred_mask] = overlay_pred[pred_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[1, 1].imshow(overlay_pred.astype(np.uint8))
    axes[1, 1].set_title('Pred Overlay (Red)', fontsize=14)
    axes[1, 1].axis('off')
    
    # 对比显示
    overlay_compare = image.copy()
    # TP: 黄色 (Red + Green)
    tp_mask = np.logical_and(gt_mask, pred_mask)
    # FP: 红色 (只有预测)
    fp_mask = np.logical_and(pred_mask, ~gt_mask)
    # FN: 绿色 (只有GT)
    fn_mask = np.logical_and(gt_mask, ~pred_mask)
    
    overlay_compare[tp_mask] = overlay_compare[tp_mask] * 0.5 + np.array([255, 255, 0]) * 0.5
    overlay_compare[fp_mask] = overlay_compare[fp_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay_compare[fn_mask] = overlay_compare[fn_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    axes[1, 2].imshow(overlay_compare.astype(np.uint8))
    
    # 添加指标文本
    metrics_text = (
        f"IoU: {metrics['iou']:.4f}\n"
        f"Dice: {metrics['dice']:.4f}\n"
        f"clDice: {metrics['cldice']:.4f}\n"
        f"TP (Yellow) | FP (Red) | FN (Green)"
    )
    axes[1, 2].set_title('Comparison\n' + metrics_text, fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(args):
    """运行推理"""
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("推理和可视化")
    print("=" * 60)
    print()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = VesselDataset(
        image_dir=config['data']['train_image_dir'],
        mask_dir=config['data'].get('train_mask_dir'),
        image_size=tuple(config['env']['image_size'])
    )
    print()
    
    # 加载 SAM2
    print("正在加载 SAM2 模型...")
    sam_generator = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=config['sam2']['use_half_precision']
    )
    print()
    
    # 加载奖励计算器
    reward_calculator = RewardCalculator(config['reward'])
    
    # 创建环境
    print("正在创建环境...")
    env = CandidateSelectionEnv(sam_generator, reward_calculator, config)
    print("✓ 环境创建完成\n")
    
    # 加载训练好的模型
    print(f"正在加载模型: {args.model_path}")
    model = PPO.load(args.model_path, env=env)
    print("✓ 模型加载完成\n")
    
    # 运行推理
    print("=" * 60)
    print("开始推理")
    print("=" * 60)
    print()
    
    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    
    # 随机选择样本或使用全部
    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    else:
        indices = range(len(dataset))
    
    results = []
    
    for idx in tqdm(indices, desc="推理进度"):
        sample = dataset[idx]
        image = sample['image']
        gt_mask = sample.get('mask', None)
        name = sample['name']
        
        if gt_mask is None:
            print(f"⚠️  跳过 {name}：无标注")
            continue
        
        # 重置环境
        options = {
            'image': image,
            'gt_mask': gt_mask
        }
        obs, info = env.reset(options=options)
        
        # 运行推理
        done = False
        step_count = 0
        max_steps = config['env']['max_steps']
        
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        # 获取最终预测
        pred_mask = env.current_mask
        
        # 计算评估指标
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        cldice = compute_cldice(pred_mask, gt_mask)
        
        metrics = {
            'name': name,
            'iou': float(iou),
            'dice': float(dice),
            'cldice': float(cldice),
            'steps': step_count
        }
        
        results.append(metrics)
        
        # 可视化
        if args.visualize:
            vis_path = vis_dir / f'{name}.png'
            visualize_result(image, gt_mask, pred_mask, metrics, str(vis_path))
    
    # 计算平均指标
    avg_metrics = {
        'avg_iou': np.mean([r['iou'] for r in results]),
        'avg_dice': np.mean([r['dice'] for r in results]),
        'avg_cldice': np.mean([r['cldice'] for r in results]),
        'avg_steps': np.mean([r['steps'] for r in results]),
        'num_samples': len(results)
    }
    
    # 保存结果
    results_file = output_dir / 'inference_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'individual_results': results,
            'average_metrics': avg_metrics
        }, f, indent=2)
    
    # 打印总结
    print()
    print("=" * 60)
    print("推理完成")
    print("=" * 60)
    print()
    print(f"样本数量: {avg_metrics['num_samples']}")
    print(f"平均 IoU: {avg_metrics['avg_iou']:.4f}")
    print(f"平均 Dice: {avg_metrics['avg_dice']:.4f}")
    print(f"平均 clDice: {avg_metrics['avg_cldice']:.4f}")
    print(f"平均步数: {avg_metrics['avg_steps']:.2f}")
    print()
    print(f"结果已保存到: {results_file}")
    if args.visualize:
        print(f"可视化已保存到: {vis_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='推理和可视化')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='训练好的模型路径'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/home/ubuntu/sam+RL/config/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/ubuntu/sam+RL/inference_results',
        help='输出目录'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help='推理样本数（0表示全部）'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='是否生成可视化'
    )
    
    args = parser.parse_args()
    run_inference(args)

