#!/usr/bin/env python3
"""生成虚拟掩膜用于测试"""
import cv2
import numpy as np
from pathlib import Path

image_dir = Path("/home/ubuntu/Segment_DATA/orgin_pic")
mask_dir = Path("/home/ubuntu/Segment_DATA/lab_pic")

print("生成虚拟掩膜...")

for img_path in image_dir.glob("*.jpg"):
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    h, w = img.shape[:2]
    
    # 创建一个简单的掩膜（中心区域）
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 4
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # 保存掩膜
    mask_path = mask_dir / f"{img_path.stem}.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"  ✓ {mask_path.name}")

print("完成！")

