"""
奖励函数：IoU、clDice、vesselness、拓扑约束等
"""
import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.filters import frangi


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    计算 IoU（Intersection over Union）
    
    Args:
        pred: 预测掩膜，bool
        gt: 真实标签，bool
    
    Returns:
        iou: IoU 分数，范围 [0, 1]
    """
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    计算 Dice 系数
    
    Args:
        pred: 预测掩膜，bool
        gt: 真实标签，bool
    
    Returns:
        dice: Dice 分数，范围 [0, 1]
    """
    intersection = np.logical_and(pred, gt).sum()
    
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    if pred_sum + gt_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(2 * intersection / (pred_sum + gt_sum))


def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    计算 clDice（centerline Dice）
    用于评估细长结构（如血管）的中心线保持程度
    
    Args:
        pred: 预测掩膜，bool
        gt: 真实标签，bool
    
    Returns:
        cldice: clDice 分数，范围 [0, 1]
    """
    # 提取中心线（骨架化）
    pred_skeleton = skeletonize(pred.astype(np.uint8))
    gt_skeleton = skeletonize(gt.astype(np.uint8))
    
    # 中心线 Dice
    # tprec: 预测中心线在 GT 掩膜内的比例
    # tsens: GT 中心线在预测掩膜内的比例
    
    pred_skeleton_sum = pred_skeleton.sum()
    gt_skeleton_sum = gt_skeleton.sum()
    
    if pred_skeleton_sum == 0 or gt_skeleton_sum == 0:
        return 0.0
    
    # Precision: 预测骨架在 GT 掩膜内的比例
    tprec = np.logical_and(pred_skeleton, gt).sum() / pred_skeleton_sum
    
    # Recall: GT 骨架在预测掩膜内的比例
    tsens = np.logical_and(gt_skeleton, pred).sum() / gt_skeleton_sum
    
    if tprec + tsens == 0:
        return 0.0
    
    cldice = 2 * tprec * tsens / (tprec + tsens)
    
    return float(cldice)


def compute_topology_penalty(mask: np.ndarray) -> float:
    """
    计算拓扑惩罚：连通组件数越多，惩罚越大
    （血管应该是连通的，过多的断裂会增加连通组件数）
    
    Args:
        mask: 掩膜，bool
    
    Returns:
        penalty: 惩罚值，范围 [0, 1]（0 表示理想，1 表示最差）
    """
    # 计算连通组件数
    num_components, _ = cv2.connectedComponents(mask.astype(np.uint8))
    num_components -= 1  # 减去背景
    
    if num_components <= 1:
        return 0.0
    
    # 归一化惩罚：假设超过 5 个组件就是非常差的情况
    penalty = min(1.0, (num_components - 1) / 5.0)
    
    return penalty


def compute_mask_quality_reward(mask: np.ndarray) -> float:
    """
    计算掩膜质量奖励（用于无原始图像的情况）
    
    评估标准：
    1. 面积合理性（30-70%覆盖）
    2. 连通性（单个连通组件最好）
    3. 形状紧凑性
    
    Args:
        mask: 二值掩膜，bool
    
    Returns:
        score: 质量分数，范围 [0, 1]
    """
    if not isinstance(mask, np.ndarray):
        return 0.0
    
    if mask.sum() == 0:
        return 0.0
    
    H, W = mask.shape
    total_pixels = H * W
    mask_uint8 = mask.astype(np.uint8)
    
    # 1. 面积合理性（30-70%覆盖）
    area_ratio = mask.sum() / total_pixels
    if 0.3 <= area_ratio <= 0.7:
        area_score = 1.0
    elif 0.2 <= area_ratio < 0.3 or 0.7 < area_ratio <= 0.8:
        area_score = 0.5
    elif 0.1 <= area_ratio < 0.2 or 0.8 < area_ratio <= 0.9:
        area_score = 0.2
    else:
        area_score = 0.0
    
    # 2. 连通性（单个连通组件最好）
    num_components, _ = label(mask_uint8)
    if num_components == 1:
        connectivity_score = 1.0
    elif num_components <= 3:
        connectivity_score = 0.7
    elif num_components <= 5:
        connectivity_score = 0.4
    else:
        connectivity_score = max(0, 1.0 - (num_components - 1) * 0.1)
    
    # 3. 形状紧凑性（不要太碎）
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 0:
        # 使用最大轮廓
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if perimeter > 0 and area > 0:
            # 紧凑度：圆形为1，越不规则越小
            compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
            shape_score = min(1.0, compactness * 2)  # 放大一点，避免过于严格
        else:
            shape_score = 0.0
    else:
        shape_score = 0.0
    
    # 综合分数
    total_score = (area_score * 0.4 + connectivity_score * 0.4 + shape_score * 0.2)
    
    # 调试输出
    print(f"      [MASK_QUALITY] 面积比:{area_ratio:.3f}(得分:{area_score:.2f}), "
          f"连通组件:{num_components}(得分:{connectivity_score:.2f}), "
          f"紧凑度:({shape_score:.2f}), 总分:{total_score:.3f}")
    
    return float(total_score)


def compute_vesselness(image: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算 Frangi vesselness 覆盖率（用于无标注数据）
    
    Args:
        image: 原始图像，uint8 或 float，可以是灰度或RGB
        mask: 掩膜，bool（可选）
    
    Returns:
        score: vesselness 覆盖分数，范围 [0, 1]
    """
    # 转换为灰度图（如果是RGB）
    if len(image.shape) == 3:
        # RGB -> 灰度
        import cv2
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    
    # 归一化图像
    if image_gray.dtype == np.uint8:
        image_norm = image_gray.astype(np.float32) / 255.0
    else:
        image_norm = image_gray.astype(np.float32)
    
    # 计算 Frangi vesselness
    # sigmas: 血管宽度范围（像素）
    vesselness = frangi(image_norm, sigmas=range(1, 10, 2), black_ridges=False)
    
    # 调试日志
    mask_sum = mask.sum() if mask is not None else 0
    print(f"      [VESSELNESS] 灰度图像形状:{image_gray.shape}, 掩膜面积:{mask_sum}")
    
    if mask is None:
        # 返回平均 vesselness
        result = float(vesselness.mean())
        print(f"      [VESSELNESS] 无掩膜模式: {result:.4f}")
        return result
    else:
        # 返回掩膜内的平均 vesselness
        if mask.sum() == 0:
            print(f"      [VESSELNESS] 掩膜为空: 0.0")
            return 0.0
        result = float(vesselness[mask].mean())
        print(f"      [VESSELNESS] 掩膜内平均: {result:.4f}")
        return result


def compute_false_positive_penalty(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    计算假阳性惩罚
    
    Args:
        pred: 预测掩膜，bool
        gt: 真实标签，bool
    
    Returns:
        penalty: 假阳性比例，范围 [0, 1]
    """
    false_positive = np.logical_and(pred, ~gt).sum()
    pred_sum = pred.sum()
    
    if pred_sum == 0:
        return 0.0
    
    return float(false_positive / pred_sum)


def compute_boundary_smoothness(mask: np.ndarray) -> float:
    """
    计算边界平滑度（边界曲率的倒数）
    
    Args:
        mask: 掩膜，bool
    
    Returns:
        smoothness: 平滑度分数，范围 [0, 1]（越大越平滑）
    """
    # 提取边界
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    
    if len(contours) == 0:
        return 0.0
    
    # 计算边界周长与面积的比值（紧凑度）
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    area = mask.sum()
    
    if area == 0:
        return 0.0
    
    # 圆的紧凑度最高：P = 2*pi*r, A = pi*r^2 => P^2 / A = 4*pi
    compactness = total_perimeter ** 2 / (4 * np.pi * area)
    
    # 归一化：紧凑度越接近 1（圆），越平滑
    smoothness = 1.0 / (1.0 + compactness)
    
    return float(smoothness)


class RewardCalculator:
    """
    奖励计算器：整合多种奖励项
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 奖励配置字典
        """
        self.use_gt = config.get('use_gt', True)
        self.iou_weight = config.get('iou_weight', 1.0)
        self.cldice_weight = config.get('cldice_weight', 0.5)
        self.topology_weight = config.get('topology_weight', 0.3)
        self.vesselness_weight = config.get('vesselness_weight', 0.5)
        self.smoothness_weight = config.get('smoothness_weight', 0.1)
        self.action_cost = config.get('action_cost', -0.01)
        self.false_positive_penalty = config.get('false_positive_penalty', -0.5)
    
    def compute_reward(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray = None,
        image: np.ndarray = None,
        prev_mask: np.ndarray = None,
        action_type: str = None
    ) -> dict:
        """
        计算综合奖励
        
        Args:
            pred_mask: 当前预测掩膜，bool
            gt_mask: 真实标签，bool（可选）
            image: 原始图像，用于计算 vesselness（可选）
            prev_mask: 上一步的掩膜，用于计算增量奖励（可选）
            action_type: 动作类型（可选）
        
        Returns:
            reward_dict: 包含总奖励和各项分奖励的字典
        """
        reward_dict = {}
        total_reward = 0.0
        
        # 动作成本
        if action_type is not None:
            reward_dict['action_cost'] = self.action_cost
            total_reward += self.action_cost
        
        # 有标注数据的奖励
        if self.use_gt and gt_mask is not None:
            # IoU
            iou = compute_iou(pred_mask, gt_mask)
            reward_dict['iou'] = iou
            total_reward += self.iou_weight * iou
            
            # clDice
            cldice = compute_cldice(pred_mask, gt_mask)
            reward_dict['cldice'] = cldice
            total_reward += self.cldice_weight * cldice
            
            # 假阳性惩罚
            fp_penalty = compute_false_positive_penalty(pred_mask, gt_mask)
            reward_dict['fp_penalty'] = fp_penalty
            total_reward += self.false_positive_penalty * fp_penalty
            
            # 增量奖励（鼓励每步带来提升）
            if prev_mask is not None:
                prev_iou = compute_iou(prev_mask, gt_mask)
                delta_iou = iou - prev_iou
                reward_dict['delta_iou'] = delta_iou
                # 增量奖励会自动体现在 iou 项中
        
        # 无标注数据的奖励（或作为辅助）
        if image is not None and (not self.use_gt or gt_mask is None):
            vesselness = compute_vesselness(image, pred_mask)
            reward_dict['vesselness'] = vesselness
            total_reward += self.vesselness_weight * vesselness
        
        # 拓扑惩罚（有无标注都可用）
        topology_penalty = compute_topology_penalty(pred_mask)
        reward_dict['topology_penalty'] = topology_penalty
        total_reward -= self.topology_weight * topology_penalty
        
        # 边界平滑度（有无标注都可用）
        smoothness = compute_boundary_smoothness(pred_mask)
        reward_dict['smoothness'] = smoothness
        total_reward += self.smoothness_weight * smoothness
        
        reward_dict['total'] = total_reward
        
        return reward_dict

