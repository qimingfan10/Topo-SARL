"""
推理脚本：使用训练好的候选选择器进行预测
"""
import argparse
import os
import yaml
import numpy as np
import cv2
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/sam+RL')

from models.sam2_wrapper import SAM2CandidateGenerator
from models.ppo_policy import create_ppo_model
from rewards.reward_functions import RewardCalculator, compute_iou, compute_dice, compute_cldice
from env.candidate_selection_env import CandidateSelectionEnv
from utils.data_loader import VesselDataset

from stable_baselines3 import PPO


def inference(args):
    """推理主函数"""
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("========================================")
    print("阶段 A：候选选择器推理")
    print("========================================\n")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = VesselDataset(
        image_dir=args.input_dir,
        mask_dir=args.gt_dir if args.gt_dir else None,
        image_size=tuple(config['env']['image_size'])
    )
    
    if len(dataset) == 0:
        raise ValueError(f"数据集为空，请检查路径: {args.input_dir}")
    
    print()
    
    # 初始化 SAM2
    print("正在加载 SAM2 模型...")
    sam_generator = SAM2CandidateGenerator(
        checkpoint=config['sam2']['checkpoint'],
        model_cfg=config['sam2']['model_cfg'],
        device=config['sam2']['device'],
        use_half_precision=config['sam2']['use_half_precision']
    )
    print()
    
    # 初始化奖励计算器
    reward_calculator = RewardCalculator(config['reward'])
    
    # 创建环境
    print("正在创建 RL 环境...")
    env = CandidateSelectionEnv(sam_generator, reward_calculator, config)
    print("✓ 环境创建完成\n")
    
    # 加载训练好的模型
    print(f"正在加载模型: {args.model}")
    model = PPO.load(args.model, env=env)
    print("✓ 模型加载完成\n")
    
    # 推理
    print("========================================")
    print("开始推理")
    print("========================================\n")
    
    results = []
    
    for idx in tqdm(range(len(dataset)), desc="推理进度"):
        sample = dataset[idx]
        image = sample['image']
        gt_mask = sample.get('mask', None)
        name = sample['name']
        
        # 重置环境
        obs, _ = env.reset(options={'image': image, 'gt_mask': gt_mask})
        
        done = False
        step = 0
        
        while not done and step < config['env']['max_steps']:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        
        # 获取最终掩膜
        pred_mask = env.current_mask
        
        # 保存预测结果
        output_path = os.path.join(args.output_dir, f"{name}.png")
        cv2.imwrite(output_path, (pred_mask * 255).astype(np.uint8))
        
        # 如果需要可视化
        if args.visualize:
            vis = env.render()
            vis_path = os.path.join(args.output_dir, f"{name}_vis.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # 如果有 GT，计算评估指标
        if gt_mask is not None:
            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            cldice = compute_cldice(pred_mask, gt_mask)
            
            results.append({
                'name': name,
                'iou': iou,
                'dice': dice,
                'cldice': cldice
            })
    
    # 打印评估结果
    if len(results) > 0:
        print("\n========================================")
        print("评估结果")
        print("========================================\n")
        
        avg_iou = np.mean([r['iou'] for r in results])
        avg_dice = np.mean([r['dice'] for r in results])
        avg_cldice = np.mean([r['cldice'] for r in results])
        
        print(f"平均 IoU:    {avg_iou:.4f}")
        print(f"平均 Dice:   {avg_dice:.4f}")
        print(f"平均 clDice: {avg_cldice:.4f}")
        
        # 保存详细结果
        if args.save_metrics:
            metrics_path = os.path.join(args.output_dir, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"平均 IoU:    {avg_iou:.4f}\n")
                f.write(f"平均 Dice:   {avg_dice:.4f}\n")
                f.write(f"平均 clDice: {avg_cldice:.4f}\n\n")
                f.write("详细结果:\n")
                for r in results:
                    f.write(f"{r['name']}: IoU={r['iou']:.4f}, Dice={r['dice']:.4f}, clDice={r['cldice']:.4f}\n")
            print(f"\n评估指标已保存: {metrics_path}")
    
    print("\n========================================")
    print("推理完成！")
    print("========================================")
    print(f"输出目录: {args.output_dir}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='候选选择器推理（阶段 A）')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/ubuntu/sam+RL/config/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='训练好的模型路径（.zip 文件）'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='输入图像目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        default=None,
        help='真实标签目录（可选，用于评估）'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='是否保存可视化结果'
    )
    parser.add_argument(
        '--save_metrics',
        action='store_true',
        help='是否保存评估指标'
    )
    
    args = parser.parse_args()
    
    inference(args)

