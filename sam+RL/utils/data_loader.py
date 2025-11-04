"""
数据加载工具
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class VesselDataset:
    """
    血管图像数据集
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Args:
            image_dir: 图像目录
            mask_dir: 掩膜目录（可选，用于有监督训练）
            image_size: 目标图像尺寸 (H, W)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size
        
        # 扫描图像文件
        self.image_files = self._scan_images()
        
        print(f"✓ 数据集加载完成")
        print(f"  图像目录: {image_dir}")
        print(f"  掩膜目录: {mask_dir if mask_dir else '无（无监督模式）'}")
        print(f"  图像数量: {len(self.image_files)}")
    
    def _scan_images(self) -> List[Path]:
        """扫描图像文件"""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(self.image_dir.glob(ext))
        
        # 过滤掉掩膜文件（文件名包含'mask'）
        image_files = [f for f in image_files if 'mask' not in f.name.lower()]
        
        image_files = sorted(image_files)
        
        return image_files
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取一个数据样本
        
        Returns:
            sample: 包含 'image', 'mask'(可选), 'name' 的字典
        """
        image_path = self.image_files[idx]
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        if image.shape[:2] != self.image_size:
            image = cv2.resize(
                image,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        sample = {
            'image': image,
            'name': image_path.stem
        }
        
        # 读取掩膜（如果有）
        if self.mask_dir is not None:
            # 尝试多种掩膜文件名格式
            possible_mask_names = [
                f"{image_path.stem}_mask.png",  # 优先匹配 xxx_mask.png
                f"{image_path.stem}.png",       # xxx.png
                f"{image_path.stem}_mask.jpg",  # xxx_mask.jpg
            ]
            
            mask_path = None
            for mask_name in possible_mask_names:
                candidate = self.mask_dir / mask_name
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path and mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # 二值化
                    mask = (mask > 127).astype(bool)
                    
                    # 调整尺寸
                    if mask.shape != self.image_size:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (self.image_size[1], self.image_size[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    sample['mask'] = mask
        
        return sample
    
    def get_random_sample(self) -> dict:
        """随机获取一个样本"""
        idx = np.random.randint(0, len(self))
        return self[idx]

