#!/usr/bin/env python3
"""
单独训练RewardNet的脚本
用于快速训练和测试RewardNet模型
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/RL4Seg3D')

import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def train_rewardnet(data_path, output_path, epochs=10, devices=1):
    """
    训练RewardNet模型
    
    Args:
        data_path: rewardDS数据目录（包含images/gt/pred子目录）
        output_path: 模型保存路径
        epochs: 训练轮数
        devices: GPU设备数量
    """
    print(f"正在训练RewardNet...")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print(f"训练轮数: {epochs}")
    
    # 检查数据目录
    data_path = Path(data_path)
    if not (data_path / "images").exists():
        raise ValueError(f"找不到images目录: {data_path}/images")
    
    # 统计数据文件数量
    image_files = list((data_path / "images").glob("*.nii.gz"))
    print(f"找到 {len(image_files)} 个训练样本")
    
    if len(image_files) < 2:
        print("\n警告: 训练样本太少！建议至少有10个样本才能有效训练RewardNet。")
        print("请先运行更多的预测来生成更多训练数据。")
        return False
    
    # 导入必要的模块
    try:
        from rl4seg3d.rewardnet.reward_unet_3d_datamodule import RewardNet3DDataModule
        from rl4seg3d.rewardnet.reward_unet_3d import Reward3DOptimizer
        from rl4seg3d.config.model.net.unet import get_unet_3d
    except Exception as e:
        print(f"导入模块失败: {e}")
        print("尝试使用简化配置...")
        # 创建简单的3D UNet
        from torch import nn
        
        class SimpleUNet3D(nn.Module):
            def __init__(self, in_channels=2, out_channels=1):
                super().__init__()
                self.patch_size = [64, 64, 2]  # 根据数据调整
                
                # 简单的编码器-解码器结构
                self.encoder = nn.Sequential(
                    nn.Conv3d(in_channels, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(32, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(32, out_channels, 3, padding=1),
                )
            
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        net = SimpleUNet3D()
    else:
        # 使用配置创建网络
        net = get_unet_3d()
    
    # 创建数据模块
    datamodule = RewardNet3DDataModule(data_path=str(data_path))
    
    # 创建模型
    model = Reward3DOptimizer(
        net=net,
        save_model_path=output_path,
        var_file=str(Path(output_path).parent / "variables.pkl")
    )
    
    # 创建logger
    logger = TensorBoardLogger(
        save_dir=str(Path(output_path).parent.parent),
        name="RewardNet_Manual_Training"
    )
    
    # 创建trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices if torch.cuda.is_available() else 1,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,  # 我们手动保存
    )
    
    # 训练
    print("\n开始训练...")
    try:
        trainer.fit(model, datamodule)
        print(f"\n✓ 训练完成！")
        print(f"模型已保存到: {output_path}")
        
        # 验证文件
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024*1024)
            print(f"模型文件大小: {file_size:.2f} MB")
            return True
        else:
            print(f"警告: 模型文件未找到: {output_path}")
            return False
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="单独训练RewardNet")
    parser.add_argument('--data-dir', default='/home/ubuntu/my_rl4seg3d_logs/3d_test',
                       help='包含rewardDS的目录')
    parser.add_argument('--output', default=None,
                       help='输出模型路径（默认：data-dir/0/rewardnet.ckpt）')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数（默认：50）')
    parser.add_argument('--devices', type=int, default=1,
                       help='GPU设备数量（默认：1）')
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir)
    data_path = data_dir / "rewardDS"
    
    if not data_path.exists():
        print(f"✗ 数据目录不存在: {data_path}")
        print("\n请先运行预测来生成训练数据：")
        print("  cd /home/ubuntu/RL4Seg3D")
        print("  python3 rl4seg3d/auto_iteration.py")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "0" / "rewardnet.ckpt"
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 训练
    if train_rewardnet(str(data_path), str(output_path), args.epochs, args.devices):
        print("\n✓ 成功！现在可以使用这个RewardNet模型进行RL训练了。")
        return 0
    else:
        print("\n✗ 训练失败，请检查错误信息。")
        return 1

if __name__ == '__main__':
    sys.exit(main())

