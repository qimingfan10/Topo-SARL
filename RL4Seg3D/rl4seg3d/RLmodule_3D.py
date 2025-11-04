import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import logging
import os
import torchio as tio
from einops import rearrange
from lightning import LightningModule
from monai.data import MetaTensor
from scipy import ndimage
from torch import Tensor
from torch.nn.functional import pad
from torchvision.transforms.functional import adjust_contrast, rotate

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from rl4seg3d.utils.Metrics import accuracy, dice_score
from rl4seg3d.utils.correctors import AEMorphoCorrector
from rl4seg3d.utils.file_utils import save_to_reward_dataset
from rl4seg3d.utils.logging_helper import log_sequence, log_video
from rl4seg3d.utils.tensor_utils import convert_to_numpy
from rl4seg3d.utils.test_metrics import full_test_metrics
from rl4seg3d.utils.Metrics import is_anatomically_valid
from rl4seg3d.utils.temporal_metrics import check_temporal_validity
from vital.metrics.camus.anatomical.utils import check_segmentation_validity


class RLmodule3D(LightningModule):

    def __init__(self, actor, reward,
                 corrector=None,
                 actor_save_path=None,
                 critic_save_path=None,
                 save_uncertainty_path=None,
                 predict_save_dir=None,
                 predict_do_model_perturb=True,
                 predict_do_img_perturb=True,
                 predict_do_corrections=True,
                 predict_do_temporal_glitches=True,
                 save_on_test=True,
                 vae_on_test=False,
                 worst_frame_thresholds=None, #{"anatomical": 0.985},
                 save_csv_after_predict=None,
                 val_batch_size=4,
                 tta=True,
                 tto='off',
                 temp_files_path='.',
                 inference=False,
                 ckpt_path=None,
                 loss=None,
                 *args: Any, **kwargs: Any) -> None:
        # Remove parameters that shouldn't be passed to parent class
        kwargs.pop('seed', None)
        kwargs.pop('ckpt_path', None)
        kwargs.pop('loss', None)
        super().__init__(*args, **kwargs)
        
        # Store parameters for later use
        self.ckpt_path = ckpt_path
        self.loss_config = loss

        self.save_hyperparameters(logger=False, ignore=["actor", "reward", "corrector"])

        self.actor = actor
        self.reward_func = reward
        if hasattr(self.reward_func, 'net'):
            self.register_module('rewardnet', self.reward_func.net)
        elif hasattr(self.reward_func, 'nets'):
            if isinstance(self.reward_func.nets, list):  # backward compatibility
                for i, n in enumerate(self.reward_func.nets):
                    self.register_module(f'rewardnet_{i}', n)
            elif isinstance(self.reward_func.nets, dict):
                for i, n in enumerate(self.reward_func.nets.values()):
                    self.register_module(f'rewardnet_{i}', n)

        self.pred_corrector = corrector
        if isinstance(self.pred_corrector, AEMorphoCorrector):
            self.register_module('correctorAE', self.pred_corrector.ae_corrector.temporal_regularization.autoencoder)

        self.predicted_rows = []

        # store predict save dir for predict_step
        self.predict_save_dir = predict_save_dir

        self.temp_files_path = Path(temp_files_path)
        if not self.temp_files_path.exists and self.trainer.global_rank == 0:
            self.temp_files_path.mkdir(parents=True, exist_ok=True)

        # for test time overfitting
        self.initial_test_params = None

    def configure_optimizers(self):
        return self.actor.get_optimizers()

    @torch.no_grad()
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        生成预测并（可选）保存为 RewardNet 数据集三件套：images/、gt/、pred/。
        要求：self.predict_save_dir 非空。
        """
        try:
            # 兼容不同的键名：'img' (训练) 或 'image' (推理)
            imgs = batch.get('img') or batch.get('image')  # [B, 1, H, W, D]
            gt = batch.get('gt', None)
            if gt is None:
                gt = batch.get('approx_gt', None)
            
            # 获取metadata：从batch或从tensor的meta属性
            meta = batch.get('image_meta_dict', {})
            if not meta and hasattr(imgs, 'meta'):
                # 推理模式下，meta信息在MetaTensor的meta属性中
                meta = imgs.meta if isinstance(imgs.meta, dict) else {}

            if imgs is None or self.predict_save_dir is None:
                return {}

            # 预测动作（分割）: 形状约为 [B, H, W, D]
            actions = self.actor.act(imgs, sample=False)
            try:
                print(f"[predict_step] batch_idx={batch_idx} imgs_shape={getattr(imgs,'shape',None)} actions_shape={getattr(actions,'shape',None)} save_dir={self.predict_save_dir}")
            except Exception:
                pass

            # 保存到磁盘
            try:
                import numpy as np
                from rl4seg3d.utils.file_utils import save_to_reward_dataset
                case_id = None
                if isinstance(meta, dict):
                    # 兼容不同的key名称
                    case_id = meta.get('case_identifier') or meta.get('filename_or_obj')

                # 构造保存目录（在根目录下建立 rewardDS 子目录，便于 RewardNet 汇集）
                save_root = (self.predict_save_dir.rstrip('/') + '/rewardDS')
                Path(save_root).mkdir(parents=True, exist_ok=True)

                # 遍历窗口批次维度
                bsz = int(actions.shape[0]) if hasattr(actions, 'shape') else 1
                for i in range(bsz):
                    img_np = imgs[i, 0].detach().cpu().numpy()
                    pred_np = actions[i].detach().cpu().numpy()
                    if gt is not None:
                        gt_np = gt[i].detach().cpu().numpy()
                    else:
                        gt_np = np.zeros_like(pred_np, dtype=pred_np.dtype)

                    # 统一为3D体素 (H, W, D) 与数值类型
                    def _to_3d(x):
                        x = np.asarray(x)
                        if x.ndim == 4 and x.shape[0] == 1:
                            x = x[0]
                        if x.ndim != 3:
                            # 尝试移动/压缩到最后三维
                            x = np.squeeze(x)
                            if x.ndim != 3:
                                raise ValueError(f"Expected 3D array after squeeze, got shape={x.shape}")
                        return x

                    img_np = _to_3d(img_np).astype(np.float32)
                    pred_np = _to_3d(pred_np).astype(np.int16)
                    gt_np = _to_3d(gt_np).astype(np.int16)

                    # 文件名：使用 dicom_uuid + 窗口索引，避免多进程冲突
                    base = case_id if case_id is not None else f"case_{batch_idx}"
                    filename = f"{base}_w{i:03d}.nii.gz"
                    try:
                        save_to_reward_dataset(save_root, filename, img_np, gt_np, pred_np)
                        print(f"Saved predict triplet -> {save_root}/(images|gt|pred)/{filename}")
                    except Exception as se:
                        print(f"Failed saving predict triplet {filename}: {se}")
            except Exception:
                # 保存失败不影响预测流程
                pass

            return {'num_saved': int(actions.shape[0]) if hasattr(actions, 'shape') else 0}
        except Exception as e:
            print(f"⚠️  Predict step error (batch {batch_idx}): {str(e)[:100]}...")
            return {}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        Test step to evaluate model performance on test dataset.
        Args:
            batch: batch containing 'img', 'gt', and optionally 'image_meta_dict'
            batch_idx: batch index
        Returns:
            Dictionary of test metrics
        """
        # Extract batch data
        b_img = batch.get('img')
        b_gt = batch.get('gt')
        meta_dict = batch.get('image_meta_dict', {})
        
        if b_img is None:
            return {}
        
        # Squeeze batch dimension if needed (test batch size is typically 1)
        if b_img.dim() == 6 and b_img.shape[0] == 1:
            # [1, W, C, H, W2, D] -> [W, C, H, W2, D]
            b_img = b_img.squeeze(0)
        if b_gt is not None and b_gt.dim() == 5 and b_gt.shape[0] == 1:
            # [1, W, H, W2, D] -> [W, H, W2, D]
            b_gt = b_gt.squeeze(0)
        
        # Get predictions using the actor (deterministic)
        y_pred = self.actor.act(b_img, sample=False)
        
        logs = {}
        
        # Calculate metrics if ground truth is available
        if b_gt is not None:
            # Ensure shapes match
            if y_pred.shape != b_gt.shape:
                # Try to align shapes
                if b_gt.dim() == 4 and y_pred.dim() == 4:
                    # Both 4D, might need resizing
                    import torch.nn.functional as F
                    b_gt_aligned = F.interpolate(
                        b_gt.unsqueeze(1).float(), 
                        size=y_pred.shape[1:], 
                        mode='nearest'
                    ).squeeze(1).long()
                    b_gt = b_gt_aligned
                elif b_gt.dim() == 5 and b_gt.shape[1] == 1:
                    # [N, 1, H, W, D] -> [N, H, W, D]
                    b_gt = b_gt.squeeze(1)
            
            # Calculate accuracy
            acc = accuracy(y_pred, b_img, b_gt)
            logs['test_acc'] = acc.mean()
            
            # Calculate dice score
            dice = dice_score(y_pred, b_gt)
            logs['test_dice'] = dice.mean()
            
            # Calculate anatomical validity
            try:
                y_pred_np = y_pred.cpu().numpy()
                anat_valid = is_anatomically_valid(y_pred_np)
                logs['test_anat_valid'] = anat_valid.mean()
            except Exception as e:
                print(f"Warning: Could not calculate anatomical validity: {e}")
            
            # Calculate rewards for logging
            try:
                rewards = self.reward_func(y_pred, b_img, b_gt)
                if isinstance(rewards, (list, tuple)):
                    for i, r in enumerate(rewards):
                        logs[f'test_reward_{i}'] = r.mean()
                else:
                    logs['test_reward'] = rewards.mean()
            except Exception as e:
                print(f"Warning: Could not calculate rewards: {e}")
        
        # Log metrics
        self.log_dict(logs, prog_bar=True, sync_dist=True)
        
        # Optionally log images (only on main process)
        if self.trainer.local_rank == 0 and batch_idx == 0:
            try:
                import random
                idx = random.randint(0, len(b_img) - 1) if len(b_img) > 1 else 0
                log_sequence(self.logger, img=b_img[idx], title='test_Image', 
                           number=batch_idx, epoch=self.current_epoch)
                if b_gt is not None:
                    log_sequence(self.logger, img=b_gt[idx].unsqueeze(0) if b_gt[idx].dim() == 3 else b_gt[idx], 
                               title='test_GroundTruth', number=batch_idx, epoch=self.current_epoch)
                log_sequence(self.logger, img=y_pred[idx].unsqueeze(0) if y_pred[idx].dim() == 3 else y_pred[idx], 
                           title='test_Prediction', number=batch_idx, epoch=self.current_epoch)
            except Exception as e:
                print(f"Warning: Could not log test images: {e}")
        
        return logs

    @torch.no_grad()  # no grad since tensors are reused in PPO's for loop
    def rollout(self, imgs: torch.tensor, gt: torch.tensor, use_gt: torch.tensor = None, sample: bool = True):
        """
            Rollout the policy over a batch of images and ground truth pairs
        Args:
            imgs: batch of images
            gt: batch of ground truth segmentation maps
            use_gt: replace policy result with ground truth (bool mask of len of batch)
            sample: whether to sample actions from distribution or determinist

        Returns:
            Actions (used for rewards, log_pobs, etc), sampled_actions (mainly for display), log_probs, rewards
        """
        logger, debug_on = _get_debug_logger()
        rank = os.environ.get("RANK", "-1")
        if debug_on:
            try:
                logger.debug(f"[rank{rank}] rollout imgs_shape={getattr(imgs,'shape',None)} gt_shape={getattr(gt,'shape',None)} use_gt_shape={getattr(use_gt,'shape',None)}")
            except Exception:
                pass

        actions = self.actor.act(imgs, sample=sample)
        
        # 获取log_probs
        _, _, log_probs, _, _, _ = self.actor.evaluate(imgs, actions)
        
        rewards = self.reward_func(actions, imgs, gt)
        if debug_on:
            try:
                logger.debug(f"[rank{rank}] rollout actions_shape={getattr(actions,'shape',None)} rewards_type={type(rewards)}")
            except Exception:
                pass

        if use_gt is not None:
            # Replace actions with ground truth where specified
            print(f"use_gt形状: {use_gt.shape}")
            print(f"gt形状: {gt.shape}")
            print(f"actions形状: {actions.shape}")
            
            # 确保gt和actions有相同的形状
            if gt.shape != actions.shape:
                print("调整gt形状以匹配actions")
                # 如果gt的尺寸不匹配，我们需要调整它
                if len(gt.shape) == 4 and len(actions.shape) == 4:
                    # 都是4D，但尺寸不同，可能需要插值或裁剪
                    import torch.nn.functional as F
                    # 转换为float进行插值，然后转回原类型
                    gt_dtype = gt.dtype
                    gt_float = gt.float()
                    gt_resized = F.interpolate(gt_float.unsqueeze(0), size=actions.shape[1:], mode='nearest').squeeze(0)
                    gt = gt_resized.to(gt_dtype)
                    print(f"调整后的gt形状: {gt.shape}")
            
            actions = torch.where(use_gt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), gt, actions)
        
        return actions, log_probs, rewards


def _get_debug_logger():
    logger = logging.getLogger("rl4seg3d.debug")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        log_path = os.environ.get("RL4SEG3D_DEBUG_LOG", "/home/ubuntu/my_rl4seg3d_logs/debug.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger, os.environ.get("RL4SEG3D_DEBUG", "0") == "1"
