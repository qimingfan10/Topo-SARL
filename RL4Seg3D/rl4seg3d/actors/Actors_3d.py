import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical
import os
import logging


class Unet3DActorCategorical(nn.Module):
    def __init__(self, net, pretrain_ckpt=None, ref_ckpt=None):
        super().__init__()
        self.net = net
        self.old_net = copy.deepcopy(self.net)

        def _valid_ckpt(p):
            return isinstance(p, str) and p.lower() != 'none' and Path(p).exists()

        if _valid_ckpt(pretrain_ckpt):
            # if starting from pretrained model, keep version of
            ckpt = torch.load(pretrain_ckpt, weights_only=False)
            # Handle both Lightning checkpoints and pure state_dicts
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                # Lightning checkpoint: extract model state_dict
                # Try different possible prefixes for actor weights
                state_dict = {}
                for prefix in ['actor.actor.net.', 'model.actor.net.', 'actor.net.']:
                    state_dict = {k.replace(prefix, ''): v 
                                 for k, v in ckpt['state_dict'].items() 
                                 if k.startswith(prefix)}
                    if state_dict:  # If we found weights with this prefix
                        break
                
                if state_dict:  # If we found actor weights
                    self.net.load_state_dict(state_dict)
                else:  # Fallback: try to load the whole state_dict directly
                    self.net.load_state_dict(ckpt['state_dict'])
            else:
                # Pure state_dict
                self.net.load_state_dict(ckpt)

            if _valid_ckpt(ref_ckpt):
                # copy to have version of initial pretrained net
                ref_ckpt_data = torch.load(ref_ckpt, weights_only=False)
                if isinstance(ref_ckpt_data, dict) and 'state_dict' in ref_ckpt_data:
                    state_dict = {}
                    for prefix in ['actor.actor.old_net.', 'model.actor.old_net.', 'actor.old_net.', 
                                   'actor.actor.net.', 'model.actor.net.', 'actor.net.']:
                        state_dict = {k.replace(prefix, ''): v 
                                     for k, v in ref_ckpt_data['state_dict'].items() 
                                     if k.startswith(prefix)}
                        if state_dict:
                            break
                    
                    if state_dict:
                        self.old_net.load_state_dict(state_dict)
                    else:
                        self.old_net.load_state_dict(ref_ckpt_data['state_dict'])
                else:
                    self.old_net.load_state_dict(ref_ckpt_data)
        # 冻结并设置评估模式，避免额外梯度与BN统计
        self.old_net.requires_grad_(False)
        self.old_net.eval()

    def forward(self, x):
        logger, debug_on = _get_debug_logger()
        rank = os.environ.get("RANK", "-1")
        if debug_on:
            try:
                logger.debug(f"[rank{rank}] Unet3DActorCategorical.forward input_shape={x.shape} dtype={x.dtype}")
            except Exception:
                pass

        # 统一输入为5D [B,C,H,W,D]
        if x.ndim == 4:
            x = x.unsqueeze(0)
        # 固定映射：从 [B,C,H,W,D] 到 Conv3d 期望的 [B,C,D,H,W]
        x_dhw = x.permute(0, 1, 4, 2, 3).contiguous()

        # 禁用autocast以使用FP32精度，避免BF16梯度underflow
        with torch.cuda.amp.autocast(enabled=False):
            net_output = self.net(x_dhw)  # [B,Classes,D,H,W]
            logits = torch.softmax(net_output, dim=1)

        # 转回 [B,Classes,H,W,D] 以与下游保持一致
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()

        if debug_on:
            try:
                first_conv_c = _first_conv_in_channels(self.net)
                logger.debug(f"[rank{rank}] actor x_dhw_shape={x_dhw.shape} logits_shape={logits.shape} first_conv_in_channels={first_conv_c}")
            except Exception:
                pass

        dist = Categorical(probs=logits.permute(0, 2, 3, 4, 1))

        if hasattr(self, "old_net"):
            with torch.no_grad():
                # 禁用autocast以使用FP32精度
                with torch.cuda.amp.autocast(enabled=False):
                    old_out = self.old_net(x_dhw)
                    old_logits = torch.softmax(old_out, dim=1)
            old_logits = old_logits.permute(0, 1, 3, 4, 2).contiguous()  # [B,Classes,H,W,D]
            old_dist = Categorical(probs=old_logits.permute(0, 2, 3, 4, 1))
        else:
            old_dist = None
        return logits, dist, old_dist


class Unet3DCritic(nn.Module):

    def __init__(self, net, pretrain_ckpt=None):
        super().__init__()
        self.net = net

        if pretrain_ckpt:
            ckpt = torch.load(pretrain_ckpt, weights_only=False)
            # Handle both Lightning checkpoints and pure state_dicts
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                # Lightning checkpoint: extract critic state_dict
                # Try different possible prefixes for critic weights
                state_dict = {}
                for prefix in ['actor.critic.net.', 'model.actor.critic.net.', 'critic.net.']:
                    state_dict = {k.replace(prefix, ''): v 
                                 for k, v in ckpt['state_dict'].items() 
                                 if k.startswith(prefix)}
                    if state_dict:  # If we found weights with this prefix
                        break
                
                if state_dict:  # If we found critic weights
                    self.net.load_state_dict(state_dict)
                else:  # Fallback: try to load the whole state_dict directly
                    self.net.load_state_dict(ckpt['state_dict'])
            else:
                # Pure state_dict
                self.net.load_state_dict(ckpt)

    def forward(self, x):
        # 统一输入为5D [B,C,H,W,D]
        if x.ndim == 4:
            x = x.unsqueeze(0)
        # 固定映射：从 [B,C,H,W,D] 到 Conv3d 期望的 [B,C,D,H,W]
        x_dhw = x.permute(0, 1, 4, 2, 3).contiguous()

        # 禁用autocast以使用FP32精度，避免BF16梯度underflow
        with torch.cuda.amp.autocast(enabled=False):
            y = self.net(x_dhw)

        # 输出转回 [B,1,H,W,D]
        if hasattr(self.net, 'deep_supervision') and self.net.deep_supervision and self.net.training:
            return [torch.sigmoid(y_).permute(0, 1, 3, 4, 2).contiguous() for y_ in y]
        return torch.sigmoid(y).permute(0, 1, 3, 4, 2).contiguous()


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

def _first_conv_in_channels(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            return m.in_channels
    return None
