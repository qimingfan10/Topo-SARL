"""
调试 vesselness 计算
"""
import sys
sys.path.insert(0, '/home/ubuntu/sam+RL')

import numpy as np
import cv2
from skimage.filters import frangi
from rewards.reward_functions import compute_vesselness
import matplotlib.pyplot as plt

# 加载一个测试图像
image_path = "/home/ubuntu/StenUNet/dataset_test/raw/sten_0262_0000.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError(f"无法加载图像: {image_path}")

print(f"图像路径: {image_path}")
print(f"图像形状: {image.shape}")
print(f"图像dtype: {image.dtype}")
print(f"图像范围: {image.min()} - {image.max()}")
print(f"图像平均值: {image.mean()}")

# 归一化
image_norm = image.astype(np.float32) / 255.0
print(f"\n归一化后范围: {image_norm.min()} - {image_norm.max()}")

# 计算 frangi vesselness
vesselness_map = frangi(image_norm, sigmas=range(1, 10, 2), black_ridges=False)
print(f"\nVesselness 图:")
print(f"  形状: {vesselness_map.shape}")
print(f"  范围: {vesselness_map.min()} - {vesselness_map.max()}")
print(f"  平均值: {vesselness_map.mean()}")
print(f"  非零像素: {(vesselness_map > 0).sum()}")
print(f"  高值像素(>0.1): {(vesselness_map > 0.1).sum()}")

# 创建一个假的掩膜（中间区域）
h, w = image.shape
mask = np.zeros((h, w), dtype=bool)
mask[h//4:3*h//4, w//4:3*w//4] = True
print(f"\n掩膜面积: {mask.sum()}")

# 计算掩膜内的 vesselness
vesselness_in_mask = vesselness_map[mask].mean()
print(f"掩膜内平均 vesselness: {vesselness_in_mask:.6f}")

# 使用自己的函数
our_result = compute_vesselness(image, mask)
print(f"我们函数的结果: {our_result:.6f}")

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(vesselness_map, cmap='hot')
axes[1].set_title(f'Vesselness Map\n(mean={vesselness_map.mean():.4f})')
axes[1].axis('off')

axes[2].imshow(mask, cmap='gray')
axes[2].set_title(f'Mask\n(area={mask.sum()})')
axes[2].axis('off')

masked_vesselness = np.zeros_like(vesselness_map)
masked_vesselness[mask] = vesselness_map[mask]
axes[3].imshow(masked_vesselness, cmap='hot')
axes[3].set_title(f'Masked Vesselness\n(mean={vesselness_in_mask:.4f})')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('/home/ubuntu/sam+RL/vesselness_debug.png', dpi=150)
print(f"\n图片已保存: /home/ubuntu/sam+RL/vesselness_debug.png")

