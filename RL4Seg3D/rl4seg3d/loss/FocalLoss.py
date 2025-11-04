"""
Focal Loss and Combined Loss implementations for addressing class imbalance
in medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    Args:
        alpha: 平衡因子，用于调整正负样本的权重 (default: 0.25)
        gamma: 聚焦参数，降低易分类样本的权重 (default: 2.0)
        reduction: 'mean', 'sum' or 'none' (default: 'mean')
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits (未经过sigmoid), 
                   shape [B, C, H, W, D] 或 [B, H, W, D]
            targets: 真实标签, shape [B, H, W, D], 值为0或1
        
        Returns:
            Focal loss值
        """
        # 如果inputs是5D [B, C, H, W, D]，取第二个通道（前景）
        if inputs.ndim == 5 and inputs.shape[1] == 2:
            inputs = inputs[:, 1]  # 取前景通道
        elif inputs.ndim == 5:
            inputs = inputs.squeeze(1)
        
        # 确保targets和inputs形状一致
        if targets.shape != inputs.shape:
            targets = targets.reshape(inputs.shape)
        
        # 计算BCE loss（使用logits）
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # 计算pt (预测概率)
        pt = torch.exp(-BCE_loss)
        
        # 计算Focal Loss
        # F_loss = alpha * (1-pt)^gamma * BCE_loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice coefficient is a measure of overlap between two samples.
    This loss is particularly useful for imbalanced datasets.
    """
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits, shape [B, C, H, W, D] 或 [B, H, W, D]
            targets: 真实标签, shape [B, H, W, D]
        
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # 如果inputs是5D [B, C, H, W, D]，取第二个通道（前景）
        if inputs.ndim == 5 and inputs.shape[1] == 2:
            inputs = inputs[:, 1]  # 取前景通道
        elif inputs.ndim == 5:
            inputs = inputs.squeeze(1)
        
        # 应用sigmoid得到概率
        inputs = torch.sigmoid(inputs)
        
        # 确保targets和inputs形状一致
        if targets.shape != inputs.shape:
            targets = targets.reshape(inputs.shape)
        
        # Flatten
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()
        
        # 计算交集
        intersection = (inputs_flat * targets_flat).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        # 返回Dice Loss (1 - Dice)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    组合Focal Loss和Dice Loss。
    
    这种组合能够：
    1. 通过Focal Loss处理类别不平衡
    2. 通过Dice Loss直接优化分割指标
    
    Args:
        focal_weight: Focal Loss的权重 (default: 0.5)
        dice_weight: Dice Loss的权重 (default: 0.5)
        focal_alpha: Focal Loss的alpha参数 (default: 0.25)
        focal_gamma: Focal Loss的gamma参数 (default: 2.0)
        dice_smooth: Dice Loss的平滑参数 (default: 1.0)
    """
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, 
                 focal_alpha=0.25, focal_gamma=2.0, dice_smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits
            targets: 真实标签
        
        Returns:
            组合loss值
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        combined = self.focal_weight * focal + self.dice_weight * dice
        
        return combined


class WeightedBCELoss(nn.Module):
    """
    加权BCE Loss，用于处理类别不平衡。
    
    Args:
        pos_weight: 正样本（前景）的权重
    """
    
    def __init__(self, pos_weight=20.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits
            targets: 真实标签
        """
        # 如果inputs是5D，处理形状
        if inputs.ndim == 5 and inputs.shape[1] == 2:
            inputs = inputs[:, 1]
        elif inputs.ndim == 5:
            inputs = inputs.squeeze(1)
        
        if targets.shape != inputs.shape:
            targets = targets.reshape(inputs.shape)
        
        # 使用加权BCE
        loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets.float(),
            pos_weight=torch.tensor([self.pos_weight]).to(inputs.device)
        )
        
        return loss

