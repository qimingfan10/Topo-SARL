#!/usr/bin/env python3
"""
简化的预测脚本 - 直接生成v118的预测
绕过Hydra配置问题
"""
import sys
sys.path.insert(0, '/home/ubuntu/RL4Seg3D')

import torch
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm

print("=" * 80)
print("🔮 生成 version_118 的预测")
print("=" * 80)

# 配置
CKPT_PATH = "/home/ubuntu/my_rl4seg3d_logs/3d_test/PPO3D_icardio3d/version_118/checkpoints/last.ckpt"
OUTPUT_DIR = Path("/home/ubuntu/RL4Seg3D/visualization_outputs/final_v118/predictions/rewardDS")
INPUT_DIR = Path("/home/ubuntu/RL4Seg3D/visualization_outputs/version_125/predictions/rewardDS/images")

# 创建输出目录
(OUTPUT_DIR / "pred").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

print(f"\n📋 配置:")
print(f"   Checkpoint: {CKPT_PATH}")
print(f"   输入目录: {INPUT_DIR}")
print(f"   输出目录: {OUTPUT_DIR}")

# 查找输入图像
test_images = list(INPUT_DIR.glob("*.nii.gz"))
if not test_images:
    print(f"\n❌ 未找到输入图像！")
    print(f"   搜索路径: {INPUT_DIR}")
    sys.exit(1)

print(f"\n✅ 找到 {len(test_images)} 个测试图像")

# 加载checkpoint
print(f"\n加载checkpoint...")
try:
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    print(f"✅ Checkpoint加载成功")
    print(f"   Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"   Global step: {ckpt.get('global_step', 'N/A')}")
except Exception as e:
    print(f"❌ Checkpoint加载失败: {e}")
    sys.exit(1)

# 从checkpoint提取state_dict
state_dict = ckpt['state_dict']

# 使用Lightning的方式加载模型
print(f"\n加载模型...")
try:
    from rl4seg3d.PPO_3d import PPO3D
    
    # 从checkpoint直接加载模型
    model = PPO3D.load_from_checkpoint(
        CKPT_PATH,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"✅ 模型加载成功")
    print(f"   设备: {device}")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 生成预测
print(f"\n{'='*80}")
print("开始预测...")
print('='*80)

results = []

with torch.no_grad():
    for i, img_file in enumerate(tqdm(test_images, desc="预测进度")):
        try:
            # 加载图像
            img_nib = nib.load(str(img_file))
            img_data = img_nib.get_fdata()
            
            # 转换为tensor [1, 1, H, W, D]
            img_tensor = torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            # 使用actor预测
            pred = model.actor.act(img_tensor, sample=False)
            
            # 转换回numpy
            if isinstance(pred, torch.Tensor):
                pred_np = pred.squeeze().cpu().numpy()
            else:
                pred_np = pred
            
            # 统计
            nonzero_ratio = (pred_np > 0).sum() / pred_np.size
            unique_vals = np.unique(pred_np)
            
            # 分类
            if pred_np.sum() == 0:
                category = "全零"
                status = "🚨"
            elif nonzero_ratio > 0.95:
                category = "全屏mask"
                status = "🚨"
            elif nonzero_ratio > 0.7:
                category = "过度分割"
                status = "⚠️ "
            elif nonzero_ratio < 0.001:
                category = "几乎无分割"
                status = "⚠️ "
            else:
                category = "正常"
                status = "✅"
            
            results.append({
                'name': img_file.name,
                'ratio': nonzero_ratio,
                'category': category,
                'status': status
            })
            
            # 保存预测
            pred_nib = nib.Nifti1Image(pred_np, img_nib.affine)
            pred_file = OUTPUT_DIR / "pred" / img_file.name
            nib.save(pred_nib, str(pred_file))
            
            # 复制原图
            import shutil
            shutil.copy(img_file, OUTPUT_DIR / "images" / img_file.name)
            
            print(f"\n{i+1}. {img_file.name}")
            print(f"   非零比例: {nonzero_ratio:.4f} ({nonzero_ratio*100:.2f}%)")
            print(f"   唯一值: {unique_vals}")
            print(f"   {status} {category}")
            
        except Exception as e:
            print(f"\n❌ 处理 {img_file.name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()

# 统计总结
print(f"\n{'='*80}")
print("统计总结")
print('='*80)

if results:
    categories = {}
    for r in results:
        cat = r['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    total = len(results)
    print(f"\n总文件数: {total}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")
    
    avg_ratio = np.mean([r['ratio'] for r in results])
    print(f"\n平均非零比例: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
    
    # 关键判断
    print(f"\n{'='*80}")
    print("🎯 修复效果评估")
    print('='*80)
    
    print(f"""
对比分析:
  修复前 (v264, pos_weight=20): 98% 过度分割  ❌
  修复后 (v118, pos_weight=2):  {avg_ratio*100:.1f}% 
    """)
    
    if avg_ratio < 0.4:
        print(f"""
✅✅✅ 修复完全成功！

过度分割从98%降到{avg_ratio*100:.1f}%！
pos_weight从20.0降到2.0显著改善了分割质量！

下一步:
  1. 生成可视化视频
  2. 计算Dice score
  3. 撰写最终报告
        """)
    elif avg_ratio < 0.7:
        print(f"""
⚠️  部分成功，有改善但仍需调整

过度分割从98%降到{avg_ratio*100:.1f}%

建议:
  1. 尝试进一步降低pos_weight到1.0
  2. 或延长训练时间
  3. 或调整其他超参数
        """)
    else:
        print(f"""
❌ 修复效果不理想

过度分割仍然很高: {avg_ratio*100:.1f}%

需要:
  1. 检查配置是否真的生效
  2. 分析reward函数的其他问题
  3. 考虑重新设计reward
        """)

print(f"\n{'='*80}")
print("完成！")
print(f"预测保存在: {OUTPUT_DIR / 'pred'}")
print('='*80)

