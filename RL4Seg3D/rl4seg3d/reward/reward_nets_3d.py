from copy import deepcopy
from typing import Union, List
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from torch import Tensor

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from rl4seg3d.reward.generic_reward import Reward

"""
Reward functions must each have pred, img, gt as input parameters
"""


class RewardUnets3D(Reward):
    def __init__(self, net, state_dict_paths, temp_factor=1, pos_weight=20.0):
        self.nets = {}
        for name, path in state_dict_paths.items():
            n = deepcopy(net)
            if path and Path(path).exists():
                n.load_state_dict(torch.load(path))
            else:
                print(f"BEWARE! You don't have a valid path for this reward net: {name}\n"
                      "Ignore if using full checkpoint file")
            n.eval()
            self.nets.update({name: n})
        self.temp_factor = temp_factor
        self.pos_weight = pos_weight  # 前景样本权重，用于处理类别不平衡
        
        # 打印RewardNet配置
        print(f"\n{'='*80}")
        print("🎁 RewardNet 配置信息 - v266修复版")
        print(f"{'='*80}")
        print(f"📊 Reward参数:")
        print(f"  - pos_weight: {pos_weight} (修复: 从20.0降到2.0，避免过度分割)")
        print(f"  - temp_factor: {temp_factor}")
        print(f"  - reward_nets: {list(state_dict_paths.keys())}")
        print(f"{'='*80}\n")

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        # 统一为5D：[B,C,H,W,D]，并构造 [B,2,H,W,D] 输入给RewardNet
        # imgs: 允许 [C,H,W,D] 或 [B,C,H,W,D]
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(0)
        if imgs.ndim != 5:
            raise ValueError(f"imgs must be 5D [B,C,H,W,D], got {imgs.shape}")

        # 仅保留图像的第一个通道
        imgs_single = imgs[:, :1, ...]

        # pred: 可能是 [B,H,W,D] 或 [B,Classes,H,W,D] 或 [H,W,D]
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)  # add batch
            pred = pred.unsqueeze(1)  # add channel
        elif pred.ndim == 4:
            pred = pred.unsqueeze(1)  # [B,1,H,W,D]
        elif pred.ndim == 5:
            if pred.shape[1] > 1:
                pred = torch.argmax(pred, dim=1, keepdim=False)  # [B,H,W,D]
                pred = pred.unsqueeze(1)  # [B,1,H,W,D]
        else:
            raise ValueError(f"pred must be 3D/4D/5D tensor, got {pred.shape}")

        # 对齐空间尺寸（最近邻插值），interpolate 需要(N,C,D,H,W)
        if imgs_single.shape[-3:] != pred.shape[-3:]:
            pred_dhw = pred.permute(0,1,4,2,3)
            target_size = (imgs_single.shape[-1], imgs_single.shape[-3], imgs_single.shape[-2])  # (D,H,W) -> careful reorder
            # imgs_single shape [B,1,H,W,D], so DHW = (D,H,W) = (-1,-3,-2)
            target_size = (imgs_single.shape[-1], imgs_single.shape[-3], imgs_single.shape[-2])
            pred_resized = F.interpolate(pred_dhw, size=target_size, mode='nearest')
            pred = pred_resized.permute(0,1,3,4,2)

        stack = torch.cat((imgs_single, pred.float()), dim=1)
        r = []
        for net in self.get_nets():
            out = net(stack)
            # assert not torch.isnan(o).any(), "NaNs in RewardNet out"
            t = self.temp_factor if len(r) < 1 else 1 # ONLY TEMPERATURE SCALE ANATOMICAL (0)
            rew = torch.sigmoid(out / t).squeeze(1)
            for i in range(rew.shape[0]):
                for j in range(rew.shape[-1]):
                    rew[i, ..., j] = rew[i, ..., j] - rew[i, ..., j].min()
                    rew[i, ..., j] = rew[i, ..., j] / rew[i, ..., j].max()
            
            # 应用pos_weight加权：给前景预测更高的reward
            # pred是二值的，值为1的位置是前景
            if self.pos_weight > 1.0 and pred is not None:
                # 创建权重mask：前景位置权重高，背景位置权重为1
                pred_binary = (pred > 0.5).float().squeeze(1)  # [B, H, W, D]
                weight_mask = torch.ones_like(pred_binary)
                weight_mask = weight_mask + (self.pos_weight - 1.0) * pred_binary
                # 应用权重到reward
                rew = rew * weight_mask
                # 重新归一化到[0, 1]
                for i in range(rew.shape[0]):
                    for j in range(rew.shape[-1]):
                        if rew[i, ..., j].max() > 0:
                            rew[i, ..., j] = rew[i, ..., j] / rew[i, ..., j].max()
            
            r += [rew]
        if len(r) > 1:
            r = [torch.minimum(r[0], r[1])]
        return r

    def get_nets(self):
        return list(self.nets.values())

    def get_reward_index(self, reward_name):
        return list(self.nets.keys()).index(reward_name)

    @torch.no_grad()
    def predict_full_sequence(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)

        self.patch_size = list([stack.shape[-3], stack.shape[-2], 4])
        self.inferer.roi_size = self.patch_size

        return [torch.sigmoid(p).squeeze(1) for p in self.predict(stack)]

    def prepare_for_full_sequence(self, batch_size=1) -> None:  # noqa: D102
        sw_batch_size = batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.get_nets()[0].patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode='gaussian',
            cache_roi_weight_map=True,
        )

    def predict(
        self, image: Union[Tensor, MetaTensor],
    ) -> List[Union[Tensor, MetaTensor]]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if np.asarray([len(self.get_nets()[i].patch_size) == 3 for i in range(len(self.get_nets()))]).all():
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = F.pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image)
                # Inverse the padding after prediction
                return [p[..., pad_len:-pad_len] for p in pred]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor],
    ) -> List[Union[Tensor, MetaTensor]]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")

        return self.sliding_window_inference(image)

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor],
    ) -> List[Union[Tensor, MetaTensor]]:
        """Inference using sliding window.

        Args:
            image: Image to predict.

        Returns:
            Predicted logits.
        """
        return [self.inferer(
            inputs=image,
            network=n,
        ) for n in self.get_nets()]
