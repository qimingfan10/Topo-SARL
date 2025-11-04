"""
检查图像内容
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载几个样本
samples = [
    "/home/ubuntu/StenUNet/dataset_test/raw/sten_0262_0000.png",
    "/home/ubuntu/StenUNet/dataset_test/raw/sten_0263_0000.png",
    "/home/ubuntu/StenUNet/dataset_test/raw/An Cong Xue(0000932433)_1-3_1_051C3E6A_frame_000028_mask.png"
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, path in enumerate(samples):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(f'{path.split("/")[-1]}\nRange: [{img.min()}, {img.max()}]\nMean: {img.mean():.1f}')
    axes[idx].axis('off')
    
    # 打印直方图信息
    hist, bins = np.histogram(img.flatten(), bins=10)
    print(f"\n{path.split('/')[-1]}:")
    print(f"  直方图: {hist}")
    print(f"  大部分像素值在: {bins[np.argmax(hist)]:.0f} - {bins[np.argmax(hist)+1]:.0f}")

plt.tight_layout()
plt.savefig('/home/ubuntu/sam+RL/image_samples.png', dpi=150)
print(f"\n图片已保存: /home/ubuntu/sam+RL/image_samples.png")

