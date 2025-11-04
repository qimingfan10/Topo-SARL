"""
SAM2 包装器：生成多个候选掩膜
"""
import sys
sys.path.insert(0, '/home/ubuntu/sam2')

import torch
import numpy as np
from typing import List, Tuple, Dict
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2CandidateGenerator:
    """
    SAM2 多候选生成器
    通过不同的提示采样和阈值生成多个候选掩膜
    """
    
    def __init__(
        self,
        checkpoint: str,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
        use_half_precision: bool = True
    ):
        """
        Args:
            checkpoint: SAM2 权重文件路径
            model_cfg: SAM2 配置文件路径（相对于 sam2 仓库根目录）
            device: 设备
            use_half_precision: 是否使用 bfloat16
        """
        self.device = device
        self.use_half_precision = use_half_precision
        
        # 构建 SAM2 模型
        # model_cfg 应该是相对路径（Hydra 会从 sam2 包中查找）
        self.sam2 = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2)
        
        # 缓存当前图像的 embedding
        self._current_image = None
        self._embedding_cached = False
        
        print(f"✓ SAM2 模型已加载: {checkpoint}")
        print(f"  设备: {device}")
        print(f"  半精度: {use_half_precision}")
    
    def set_image(self, image: np.ndarray):
        """
        设置图像并预计算 embedding（缓存以便多次调用）
        
        Args:
            image: RGB 图像，shape (H, W, 3)，uint8
        """
        self._current_image = image
        
        # 预计算 embedding
        with torch.inference_mode():
            if self.use_half_precision:
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    self.predictor.set_image(image)
            else:
                self.predictor.set_image(image)
        
        self._embedding_cached = True
    
    def generate_candidates(
        self,
        points: np.ndarray = None,
        point_labels: np.ndarray = None,
        boxes: np.ndarray = None,
        mask_thresholds: List[float] = [0.0, 0.5, 1.0],
        multimask_output: bool = True
    ) -> List[Dict]:
        """
        生成多个候选掩膜
        
        Args:
            points: 提示点，shape (N, 2)，坐标 (x, y)
            point_labels: 点标签，1=正点，0=负点
            boxes: 提示框，shape (N, 4)，坐标 (x1, y1, x2, y2)
            mask_thresholds: 掩膜二值化阈值列表
            multimask_output: 是否输出多个候选（SAM2 内部）
        
        Returns:
            candidates: 候选列表，每个元素包含：
                - 'mask': 二值掩膜 (H, W)，bool
                - 'score': 置信度分数
                - 'logits': 原始 logits (256, 256)
                - 'threshold': 使用的阈值
        """
        if not self._embedding_cached:
            raise RuntimeError("请先调用 set_image() 设置图像")
        
        candidates = []
        
        with torch.inference_mode():
            if self.use_half_precision:
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    masks, scores, logits = self.predictor.predict(
                        point_coords=points,
                        point_labels=point_labels,
                        box=boxes[0] if boxes is not None and len(boxes) > 0 else None,
                        multimask_output=multimask_output
                    )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box=boxes[0] if boxes is not None and len(boxes) > 0 else None,
                    multimask_output=multimask_output
                )
        
        # masks: (N, H, W)，bool
        # scores: (N,)，float
        # logits: (N, 256, 256)，float
        
        # 如果 multimask_output=True，SAM2 会返回 3 个候选
        # 我们对每个候选再应用不同的阈值，进一步增加候选数
        for i in range(len(masks)):
            for threshold in mask_thresholds:
                # 从 logits 重新生成掩膜
                if threshold == 0.0:
                    # 使用 SAM2 默认输出
                    mask = masks[i]
                else:
                    # 对 logits 应用不同阈值
                    # 需要先上采样到原始尺寸
                    mask_logits = torch.from_numpy(logits[i]).to(self.device)
                    mask_logits_upscaled = torch.nn.functional.interpolate(
                        mask_logits.unsqueeze(0).unsqueeze(0),
                        size=self._current_image.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    )[0, 0]
                    mask = (mask_logits_upscaled > threshold).cpu().numpy()
                
                candidates.append({
                    'mask': mask.astype(bool),
                    'score': float(scores[i]),
                    'logits': logits[i],
                    'threshold': threshold,
                    'sam_index': i  # 原始 SAM2 输出的索引
                })
        
        # 调试日志：记录生成的候选数和质量
        if len(candidates) > 0:
            avg_area = np.mean([c['mask'].sum() for c in candidates])
            avg_score = np.mean([c['score'] for c in candidates])
            print(f"    生成候选数: {len(candidates)}, 平均面积: {avg_area:.0f}, 平均分数: {avg_score:.3f}")
        
        return candidates
    
    def sample_random_points(
        self,
        num_points: int = 5,
        uncertainty_map: np.ndarray = None,
        exclude_mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在图像中采样随机正点（用于初始候选生成或探索）
        
        Args:
            num_points: 采样点数
            uncertainty_map: 不确定性图 (H, W)，用于引导采样（可选）
            exclude_mask: 排除区域掩膜 (H, W)，bool（可选）
        
        Returns:
            points: (N, 2)，坐标 (x, y)
            labels: (N,)，全为 1（正点）
        """
        if self._current_image is None:
            raise RuntimeError("请先调用 set_image()")
        
        h, w = self._current_image.shape[:2]
        
        if uncertainty_map is not None:
            # 根据不确定性采样（概率加权）
            uncertainty_map = uncertainty_map.copy()
            if exclude_mask is not None:
                uncertainty_map[exclude_mask] = 0
            
            # 归一化为概率分布
            prob = uncertainty_map.flatten()
            prob_sum = prob.sum()
            if prob_sum > 0:
                prob = prob / prob_sum
            else:
                # 如果全为0，使用均匀分布
                prob = np.ones_like(prob) / len(prob)
            
            # 采样
            indices = np.random.choice(len(prob), size=num_points, p=prob, replace=False)
            y_coords = indices // w
            x_coords = indices % w
        else:
            # 均匀随机采样
            valid_mask = np.ones((h, w), dtype=bool)
            if exclude_mask is not None:
                valid_mask &= ~exclude_mask
            
            valid_coords = np.argwhere(valid_mask)  # (N, 2)，格式 (y, x)
            if len(valid_coords) < num_points:
                num_points = len(valid_coords)
            
            sampled_indices = np.random.choice(len(valid_coords), size=num_points, replace=False)
            sampled_coords = valid_coords[sampled_indices]
            y_coords = sampled_coords[:, 0]
            x_coords = sampled_coords[:, 1]
        
        points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
        labels = np.ones(num_points, dtype=np.int32)
        
        return points, labels

