#!/usr/bin/env python3
"""调试预测脚本"""
import sys
sys.path.insert(0, '/home/ubuntu/RL4Seg3D')

import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig

# 设置工作目录
import os
os.chdir('/home/ubuntu/RL4Seg3D/rl4seg3d')

@hydra.main(version_base="1.3", config_path="/home/ubuntu/RL4Seg3D/rl4seg3d/config", config_name="predict3d")
def main(cfg: DictConfig):
    print("=" * 80)
    print("🔍 调试信息")
    print("=" * 80)
    
    # 检查配置
    print(f"\n📋 配置信息:")
    print(f"  input_path: {cfg.input_path}")
    print(f"  output_path: {cfg.output_path}")
    print(f"  ckpt_path: {cfg.ckpt_path}")
    print(f"  model.predict_save_dir: {cfg.model.get('predict_save_dir', 'NOT SET')}")
    
    # 实例化模型
    print(f"\n🔨 实例化模型...")
    model = hydra.utils.instantiate(cfg.model)
    print(f"  模型类型: {type(model).__name__}")
    print(f"  predict_save_dir 属性: {getattr(model, 'predict_save_dir', 'NOT SET')}")
    
    # 检查是否有 predict_step
    print(f"\n🔍 检查方法:")
    print(f"  has predict_step: {hasattr(model, 'predict_step')}")
    if hasattr(model, 'predict_step'):
        import inspect
        sig = inspect.signature(model.predict_step)
        print(f"  predict_step signature: {sig}")
    
    # 加载 checkpoint
    print(f"\n📦 加载 checkpoint...")
    ckpt = torch.load(cfg.ckpt_path, weights_only=False)
    print(f"  checkpoint keys: {list(ckpt.keys())[:5]}")
    
    # 加载到模型
    if ckpt.get("pytorch-lightning_version"):
        print(f"  Lightning checkpoint 版本: {ckpt.get('pytorch-lightning_version')}")
        # 检查 state_dict 中的 predict_save_dir
        if 'hyper_parameters' in ckpt:
            hyper = ckpt['hyper_parameters']
            print(f"  hyper_parameters keys: {list(hyper.keys())}")
            print(f"  predict_save_dir in hyper: {hyper.get('predict_save_dir', 'NOT FOUND')}")
    
    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)

if __name__ == '__main__':
    main()

