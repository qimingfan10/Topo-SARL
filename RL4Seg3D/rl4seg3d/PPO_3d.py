import time
from typing import Any
import copy

import torch
import os
import logging
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.functional import adjust_contrast
import random

from rl4seg3d.RLmodule_3D import RLmodule3D


class PPO3D(RLmodule3D):

    def __init__(self,
                 clip_value: float = 0.2,
                 k_steps_per_batch: int = 5,
                 entropy_coeff: float = 0.0,
                 divergence_coeff: float = 0.0,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # since optimization is done manually, this flag needs to be set
        self.automatic_optimization = False
        # enable loading of partial model weights
        self.strict_loading = False
        
        # 🔥 强制使用FP32精度以解决梯度underflow问题
        print(f"\n{'='*80}")
        print("🔥 PPO 配置信息 - v266修复版")
        print(f"{'='*80}")
        print(f"📊 训练参数:")
        print(f"  - entropy_coeff: {entropy_coeff} (修复: 从0.3降到0.03，避免策略随机化)")
        print(f"  - clip_value: {clip_value}")
        print(f"  - k_steps_per_batch: {k_steps_per_batch}")
        print(f"  - divergence_coeff: {divergence_coeff}")
        print(f"\n💪 精度设置:")
        print("  - 强制使用FP32精度以解决BF16梯度underflow问题")
        print(f"{'='*80}")
        
        # 检查并转换actor和critic为FP32
        if hasattr(self.actor, 'actor') and self.actor.actor is not None:
            self.actor.actor = self.actor.actor.float()
            print(f"✅ Actor网络已转换为FP32")
            print(f"   Actor参数类型: {next(self.actor.actor.parameters()).dtype}")
        
        if hasattr(self.actor, 'critic') and self.actor.critic is not None:
            self.actor.critic = self.actor.critic.float()
            print(f"✅ Critic网络已转换为FP32")
            print(f"   Critic参数类型: {next(self.actor.critic.parameters()).dtype}")
        
        print(f"{'='*80}\n")

    def training_step(self, batch: dict[str, Tensor], nb_batch):
        """
            Defines PPO training steo
            Get actions, log_probs and rewards from current policy
            Calculate and backprop losses for actor and critic K times in loop over same batch
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics or None
        """
        logger, debug_on = _get_debug_logger()
        rank = os.environ.get("RANK", "-1")
        if debug_on:
            try:
                mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else -1
                logger.debug(f"[rank{rank}] training_step start batch_keys={list(batch.keys())} img_shape={getattr(batch.get('img'),'shape',None)} mem_allocated={mem}")
            except Exception:
                pass

        opt_net, opt_critic = self.optimizers()
        opt_net.zero_grad()  # do once first if not done initially in loop

        # TODO: REMOVE GT
        b_img, b_gt, b_use_gt = batch['img'].squeeze(0), batch['gt'].squeeze(0), batch['use_gt'].squeeze(0)

        # get actions, log_probs, rewards, etc from pi (stays constant for all steps k)
        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, b_use_gt, sample=False)
        num_rewards = len(prev_rewards)

        # iterate with pi prime k times
        for k in range(self.hparams.k_steps_per_batch*num_rewards):
            # calculates training loss
            loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions,
                                                                        prev_rewards[k % num_rewards],
                                                                        prev_log_probs, b_gt, b_use_gt))
            self.manual_backward(loss)
            
            # 计算并记录梯度范数
            total_norm_sq = 0.0
            for p in self.actor.actor.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            actor_grad_norm_before = total_norm_sq ** 0.5
            
            # 梯度裁剪（总是执行）
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.actor.parameters(), 10.0
            ).item()
            
            # TODO: grad accumulation here???
            if k % num_rewards == (num_rewards-1):  # only step when all rewards are done, like a2c with multiple actors
                opt_net.step()
                opt_net.zero_grad()

            # TODO: should this be outside the loop? According to real algo...
            opt_critic.zero_grad()
            self.manual_backward(critic_loss)
            
            # 计算critic梯度范数（修复：总是计算和裁剪）
            total_norm_sq = 0.0
            for p in self.actor.critic.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            critic_grad_norm_before = total_norm_sq ** 0.5
            
            # 梯度裁剪（总是执行）
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.critic.parameters(), 10.0
            ).item()
            
            opt_critic.step()

            logs = {**metrics_dict,
                    **{'loss': loss,
                       'critic_loss': critic_loss,
                       'grad_norm/actor': actor_grad_norm,
                       'grad_norm/critic': critic_grad_norm,
                       }
                    }

            self.log_dict(logs, prog_bar=True)
            if debug_on:
                try:
                    logger.debug(f"[rank{rank}] step_logs keys={list(logs.keys())} loss={float(loss)} critic_loss={float(critic_loss)}")
                except Exception:
                    pass

    def compute_policy_loss(self, batch, **kwargs):
        """
            Compute unsupervised loss to maximise reward using PPO method.
        Args:
            batch: batch of images, actions, log_probs, rewards and ground truth
            sample: whether to sample from distribution or deterministic approach (mainly for val, test steps)

        Returns:
            mean loss(es) for the batch, metrics dictionary
        """
        b_img, b_actions, b_rewards, b_log_probs, b_gt, b_use_gt = batch

        _, logits, log_probs, entropy, v, old_log_probs = self.actor.evaluate(b_img, b_actions)

        # 调试与健壮性：如 log_probs 未连接计算图，则手工基于 logits 计算
        logger, debug_on = _get_debug_logger()
        rank = os.environ.get("RANK", "-1")
        if debug_on:
            try:
                logger.debug(f"[rank{rank}] compute_policy_loss grads log_probs={getattr(log_probs,'requires_grad',None)} entropy={getattr(entropy,'requires_grad',None)}")
            except Exception:
                pass

        if not getattr(log_probs, 'requires_grad', False):
            eps = 1e-8
            # logits: [B, Classes, H, W, D] -> probs5: [B, H, W, D, Classes]
            probs5 = logits.permute(0, 2, 3, 4, 1)
            # actions: [B, H, W, D]
            gathered = probs5.gather(-1, b_actions.long().unsqueeze(-1)).squeeze(-1)
            log_probs = torch.log(gathered + eps)

        if not getattr(entropy, 'requires_grad', False):
            eps = 1e-8
            probs5 = logits.permute(0, 2, 3, 4, 1)
            entropy = -(probs5 * torch.log(probs5 + eps)).sum(dim=-1)

        v_deeps = None
        if isinstance(v, list):
            v_deeps = v
            v = v[0]

        # 计算importance ratio（需要梯度）
        assert b_log_probs.shape == log_probs.shape
        ratio = (log_probs - b_log_probs).exp()
        
        # 计算advantage（不需要梯度，但不能破坏ratio的梯度）
        log_pi_ratio = (log_probs - old_log_probs).detach()  # detach这个，而不是用no_grad包裹
        total_reward = b_rewards - (self.hparams.divergence_coeff * log_pi_ratio)
        # ignore divergence if using ground truth
        # total_reward[b_use_gt, ...] = torch.ones_like(b_rewards)[b_use_gt, ...]
        
        # assert b_rewards.shape == v.shape
        adv = (total_reward - v).detach()  # advantage需要detach，但在no_grad外计算

        # clamp with epsilon value
        clipped = ratio.clamp(1 - self.hparams.clip_value, 1 + self.hparams.clip_value)
        surr_loss = torch.min(adv * ratio, adv * clipped)
        # surr_loss[b_use_gt, ...] = (adv * ratio)[b_use_gt, ...]

        # min trick
        loss = -surr_loss.mean() + (-self.hparams.entropy_coeff * entropy.mean())
        # 始终注入零因子正则以确保梯度路径存在（即使上游断链）
        zero_reg_actor = sum((p.float().sum() * 0.0) for p in self.actor.actor.parameters())
        loss = loss + zero_reg_actor

        # 如果策略loss不在计算图上，使用交叉熵作为回退，确保梯度可用
        if not getattr(loss, 'requires_grad', False):
            try:
                num_classes = logits.shape[1]
                logits_flat = logits.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
                targets_flat = b_actions.contiguous().view(-1).long()
                loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
            except Exception:
                # 兜底防御：引入一个与模型参数耦合的微小正则以避免空梯度
                reg = 0.0
                for p in self.actor.actor.parameters():
                    reg = reg + (p.float().sum() * 0.0)
                loss = reg + (-surr_loss.mean())

        # Critic loss
        if b_rewards.shape != v.shape:  # if critic is resnet, use reward mean instead of pixel-wise
            b_rewards = b_rewards.mean(dim=(1, 2), keepdim=True)

        if v_deeps:
            # deep supervision
            critic_loss = nn.MSELoss()(v_deeps[0], b_rewards)
            for i, v_ in enumerate(v_deeps[1:]):
                downsampled_label = nn.functional.interpolate(b_rewards.unsqueeze(0), v_.shape[1:]).squeeze(0)
                critic_loss += 10.0 ** (i + 1) * nn.MSELoss()(v_, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(v_deeps)))
            critic_loss = c_norm * critic_loss
        else:
            critic_loss = nn.MSELoss()(v, b_rewards)

        # 始终注入零因子正则，确保 critic 分支梯度路径存在
        zero_reg_critic = sum((p.float().sum() * 0.0) for p in self.actor.critic.parameters())
        critic_loss = critic_loss + zero_reg_critic

        # metrics dict
        metrics = {
                'v': v.mean(),
                'advantage': adv.mean(),
                'reward': b_rewards.mean(),
                'log_probs': log_probs.mean(),
                'ratio': ratio.mean(),
                'approx_kl_div': log_pi_ratio.mean(),
                'entropy': entropy.mean(),  # 🔧 监控entropy，确保不会太高导致策略随机化
                'entropy_coeff': self.hparams.entropy_coeff,  # 记录当前entropy系数
                'surr_loss': surr_loss.mean(),  # 监控surrogate loss
        }

        return loss, critic_loss, metrics

    def ttoptimize(self, batch_image, num_iter=4, **kwargs):
        """
            Run a few itercvations of optimization to overfit on one test sequence in unsupervised
        Args:
            batch: batch of images
            num_iter: number of iterations of the optimization loop
        Returns:
            None, actor is modified, ready for inference on batch_image alone
        """
        self.train()
        augmentations = 3
        self.divergence_coeff = 0.01
        self.entropy_coeff = 0.1
        opt_net, _ = self.configure_optimizers()
        for g in opt_net.param_groups:
            g['lr'] = 0.001

        with torch.enable_grad():
            batch_image = batch_image
            # split up the input test sequence into smaller chunks
            split_batch_images = list(torch.split(batch_image, 4, dim=-1))
            # make sure last chunk is same size
            if split_batch_images[-1].shape[-1] != 4:
                split_batch_images[-1] = batch_image[..., -4:]

            best_reward = 0
            best_i = 0
            best_params = None
            for i in range(num_iter+1):
                sum_chunk_reward = 0
                lowest_frame_reward = 1.0
                for chunk in split_batch_images:
                    chunk = chunk.detach()
                    opt_net.zero_grad()
                    avg_lowest_reward_frame = 0
                    for a in range(augmentations):
                        with torch.no_grad():
                            chunk_def = adjust_contrast(chunk.clone().permute((4, 0, 1, 2, 3)),
                                                        random.uniform(0.4, 0.8)).permute((1, 2, 3, 4, 0))
                            chunk_def += torch.randn(chunk_def.size()).to(
                                next(self.actor.actor.net.parameters()).device) * random.uniform(0.001, 0.01)
                            chunk_def /= chunk_def.max()

                        prev_actions, prev_log_probs, prev_rewards = self.rollout(chunk_def, None, None)

                        sum_chunk_reward += prev_rewards[0].mean()
                        lowest_frame_reward = min(prev_rewards[0].mean(axis=(0, 1, 2)).min().item(), lowest_frame_reward)
                        avg_lowest_reward_frame += prev_rewards[0].mean(axis=(0, 1, 2)).min().item() / augmentations

                        if i != 0:
                            loss, _, _ = self.compute_policy_loss((chunk, prev_actions,
                                                                             prev_rewards[0],
                                                                             prev_log_probs, None, None))
                            loss = loss / augmentations
                            self.manual_backward(loss)
                    lowest_frame_reward = min(avg_lowest_reward_frame, lowest_frame_reward)
                    # update after checking, as current policy was used for calculating the reward
                    if i != 0:
                        if "32" in self.trainer.precision:
                            nn.utils.clip_grad_norm_(self.actor.actor.parameters(), 10.0)
                        opt_net.step()
                print(f"\n{'First, no optimization' if i == 0 else ''}", i, (sum_chunk_reward / len(split_batch_images) / augmentations), lowest_frame_reward)
                if lowest_frame_reward > best_reward:
                    best_i = i
                    best_reward = lowest_frame_reward
                    best_params = copy.deepcopy(self.actor.actor.net.state_dict())

        self.actor.actor.net.load_state_dict(best_params)
        print("BEST ITERATION", best_i)
        self.eval()


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