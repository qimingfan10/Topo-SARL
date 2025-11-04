"""
RL 环境：候选选择环境
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple

import sys
sys.path.insert(0, '/home/ubuntu/sam+RL')

from models.sam2_wrapper import SAM2CandidateGenerator
from rewards.reward_functions import RewardCalculator


class CandidateSelectionEnv(gym.Env):
    """
    候选选择环境（阶段 A：最小可行原型）
    
    动作空间：
    - 0 ~ N-1: 选择第 i 个候选
    - N: 采样新提示点并生成新候选
    - N+1: 融合当前选中的候选
    - N+2: 终止
    
    观察空间：
    - 候选特征向量（每个候选的统计特征）
    - 当前累积掩膜的统计特征
    - 步数信息
    """
    
    metadata = {'render.modes': ['rgb_array']}
    
    def __init__(
        self,
        sam_generator: SAM2CandidateGenerator,
        reward_calculator: RewardCalculator,
        config: dict
    ):
        """
        Args:
            sam_generator: SAM2 候选生成器
            reward_calculator: 奖励计算器
            config: 环境配置
        """
        super().__init__()
        
        self.sam_generator = sam_generator
        self.reward_calculator = reward_calculator
        self.config = config
        
        self.max_steps = config['env']['max_steps']
        self.image_size = tuple(config['env']['image_size'])
        self.obs_dim = config['env']['observation_features']
        
        self.num_initial_points = config['candidates']['num_initial_points']
        self.mask_thresholds = config['candidates']['mask_thresholds']
        self.multimask_output = config['candidates']['multimask_output']
        
        # 动作空间：固定大小，包括特殊动作
        # 0-9: 选择前10个候选之一
        # 10: 采样新提示点
        # 11: 融合当前选中的候选
        # 12: 终止
        self.max_select_candidates = 10
        self.action_sample_new = 10
        self.action_merge = 11
        self.action_terminate = 12
        self.action_space = spaces.Discrete(13)
        
        # 观察空间：特征向量
        # 包含：候选特征（每个候选的统计） + 全局特征 + 元信息
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        # 状态变量
        self.current_image = None
        self.current_gt_mask = None
        self.current_mask = None  # 累积掩膜
        self.candidates = []
        self.selected_indices = []  # 已选择的候选索引
        self.step_count = 0
        self.done = False
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项，可包含 'image' 和 'gt_mask'
        
        Returns:
            observation: 初始观察
            info: 额外信息
        """
        super().reset(seed=seed)
        
        if options is None:
            options = {}
        
        # 设置当前图像和标签
        self.current_image = options.get('image')
        self.current_gt_mask = options.get('gt_mask')
        
        if self.current_image is None:
            raise ValueError("必须在 reset() 时通过 options 提供 'image'")
        
        # 调整图像尺寸
        if self.current_image.shape[:2] != self.image_size:
            self.current_image = cv2.resize(
                self.current_image,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        if self.current_gt_mask is not None and self.current_gt_mask.shape != self.image_size:
            self.current_gt_mask = cv2.resize(
                self.current_gt_mask.astype(np.uint8),
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # 设置图像到 SAM2
        self.sam_generator.set_image(self.current_image)
        
        # 生成初始候选
        self._generate_initial_candidates()
        
        # 初始化累积掩膜（全零）
        self.current_mask = np.zeros(self.image_size, dtype=bool)
        
        # 重置状态
        self.selected_indices = []
        self.step_count = 0
        self.done = False
        
        # 计算初始观察
        obs = self._get_observation()
        info = {'candidates': len(self.candidates)}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作
        
        Args:
            action: 动作索引
        
        Returns:
            observation: 新观察
            reward: 奖励
            terminated: 是否终止（成功或失败）
            truncated: 是否截断（超时）
            info: 额外信息
        """
        if self.done:
            raise RuntimeError("环境已终止，请调用 reset()")
        
        self.step_count += 1
        
        # 解析动作
        num_candidates = len(self.candidates)
        
        action_type = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if action < self.max_select_candidates:
            # 选择候选（如果候选存在）
            if action < num_candidates:
                action_type = 'select_candidate'
                selected_idx = action
                self.selected_indices.append(selected_idx)
                info['action'] = f'select_{selected_idx}'
            else:
                # 候选不足，视为无效选择，给小惩罚
                action_type = 'invalid_select'
                info['action'] = 'invalid_select'
                reward = -0.01
            
        elif action == self.action_sample_new:
            # 采样新提示点
            action_type = 'sample_new_prompt'
            self._sample_new_candidates()
            info['action'] = 'sample_new'
            info['new_candidates'] = len(self.candidates)
            
        elif action == self.action_merge:
            # 融合当前选中的候选
            action_type = 'merge_current'
            prev_mask = self.current_mask.copy()
            prev_area = prev_mask.sum()
            
            self._merge_selected_candidates()
            info['action'] = 'merge'
            info['merged_count'] = len(self.selected_indices)
            
            # 计算当前掩膜面积
            current_area = self.current_mask.sum()
            info['mask_area'] = int(current_area)
            info['area_increase'] = int(current_area - prev_area)
            
            # 计算奖励
            reward_dict = self.reward_calculator.compute_reward(
                pred_mask=self.current_mask,
                gt_mask=self.current_gt_mask,
                image=self._get_grayscale_image(),
                prev_mask=prev_mask,
                action_type=action_type
            )
            reward = reward_dict['total']
            info.update(reward_dict)
            
            # 基础merge奖励（鼓励merge动作）
            merge_bonus = 0.05
            reward += merge_bonus
            info['merge_bonus'] = merge_bonus
            
            # 如果merge后有内容，额外奖励
            if current_area > 0:
                area_bonus = 0.1
                reward += area_bonus
                info['area_bonus'] = area_bonus
            
            # 额外奖励：鼓励掩膜增长（进度奖励）
            if current_area > prev_area:
                progress_reward = 0.01 * (current_area - prev_area) / (512 * 512)
                reward += progress_reward
                info['progress_reward'] = progress_reward
            
        elif action == self.action_terminate:
            # 终止
            action_type = 'terminate'
            info['action'] = 'terminate'
            terminated = True
            
            # 惩罚过早终止
            if self.step_count < 5:
                early_penalty = -0.15
                reward += early_penalty
                info['early_terminate_penalty'] = early_penalty
            elif self.step_count < 8:
                early_penalty = -0.08
                reward += early_penalty
                info['early_terminate_penalty'] = early_penalty
            
            # 最终奖励
            reward_dict = self.reward_calculator.compute_reward(
                pred_mask=self.current_mask,
                gt_mask=self.current_gt_mask,
                image=self._get_grayscale_image(),
                prev_mask=None,
                action_type=action_type
            )
            reward += reward_dict['total']
            info.update(reward_dict)
        
        else:
            raise ValueError(f"无效动作: {action}")
        
        # 检查是否超时
        if self.step_count >= self.max_steps:
            truncated = True
        
        self.done = terminated or truncated
        
        # 添加当前IoU到info（用于追踪）
        if self.current_gt_mask is not None:
            from rewards.reward_functions import compute_iou
            current_iou = compute_iou(self.current_mask, self.current_gt_mask)
            info['current_iou'] = current_iou
            
            # 如果episode结束，记录最终IoU
            if self.done:
                info['final_iou'] = current_iou
        
        # 记录最终掩膜面积
        if self.done:
            info['final_area'] = int(self.current_mask.sum())
        
        # 记录步数
        info['step_count'] = self.step_count
        
        # 计算新观察
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _generate_initial_candidates(self):
        """生成初始候选（随机采样点）"""
        points, labels = self.sam_generator.sample_random_points(
            num_points=self.num_initial_points
        )
        
        self.candidates = self.sam_generator.generate_candidates(
            points=points,
            point_labels=labels,
            mask_thresholds=self.mask_thresholds,
            multimask_output=self.multimask_output
        )
    
    def _sample_new_candidates(self):
        """采样新的提示点并生成候选"""
        # 基于不确定性采样
        uncertainty_map = self._compute_uncertainty_map()
        
        points, labels = self.sam_generator.sample_random_points(
            num_points=1,
            uncertainty_map=uncertainty_map,
            exclude_mask=self.current_mask
        )
        
        new_candidates = self.sam_generator.generate_candidates(
            points=points,
            point_labels=labels,
            mask_thresholds=self.mask_thresholds,
            multimask_output=self.multimask_output
        )
        
        self.candidates.extend(new_candidates)
    
    def _merge_selected_candidates(self):
        """融合选中的候选到累积掩膜"""
        print(f"    [MERGE] 选中候选数: {len(self.selected_indices)}")
        prev_area = self.current_mask.sum()
        
        for idx in self.selected_indices:
            if idx < len(self.candidates):
                candidate_mask = self.candidates[idx]['mask']
                cand_area = candidate_mask.sum()
                print(f"      候选{idx}: 面积={cand_area}")
                self.current_mask = np.logical_or(self.current_mask, candidate_mask)
        
        final_area = self.current_mask.sum()
        print(f"    [MERGE] 融合前: {prev_area}, 融合后: {final_area}, 增加: {final_area-prev_area}")
        
        # 清空已选择列表
        self.selected_indices = []
    
    def _compute_uncertainty_map(self) -> np.ndarray:
        """
        计算不确定性图（基于候选的一致性）
        
        Returns:
            uncertainty_map: (H, W)，归一化到 [0, 1]
        """
        if len(self.candidates) == 0:
            return np.ones(self.image_size, dtype=np.float32)
        
        # 统计每个像素在候选中出现的频率
        vote_map = np.zeros(self.image_size, dtype=np.float32)
        for cand in self.candidates:
            vote_map += cand['mask'].astype(np.float32)
        
        vote_map /= len(self.candidates)
        
        # 不确定性 = 熵（二值）= -p*log(p) - (1-p)*log(1-p)
        # 当 p=0.5 时不确定性最大
        uncertainty = -vote_map * np.log(vote_map + 1e-8) - (1 - vote_map) * np.log(1 - vote_map + 1e-8)
        uncertainty = uncertainty / np.log(2)  # 归一化到 [0, 1]
        
        return uncertainty
    
    def _get_observation(self) -> np.ndarray:
        """
        构造观察向量
        
        Returns:
            obs: 特征向量，shape (obs_dim,)
        """
        features = []
        
        # 1. 候选统计特征（取前 5 个候选）
        max_cands_to_encode = 5
        for i in range(max_cands_to_encode):
            if i < len(self.candidates):
                cand = self.candidates[i]
                # 特征：面积、分数、阈值、与当前掩膜的重叠度
                area = cand['mask'].sum() / (self.image_size[0] * self.image_size[1])
                score = cand['score']
                threshold = cand['threshold']
                overlap = np.logical_and(cand['mask'], self.current_mask).sum() / (cand['mask'].sum() + 1e-8)
                features.extend([area, score, threshold, overlap])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 2. 当前掩膜统计
        current_area = self.current_mask.sum() / (self.image_size[0] * self.image_size[1])
        num_components, _ = cv2.connectedComponents(self.current_mask.astype(np.uint8))
        num_components -= 1
        features.extend([current_area, num_components])
        
        # 3. 不确定性统计
        uncertainty_map = self._compute_uncertainty_map()
        uncertainty_mean = uncertainty_map.mean()
        uncertainty_max = uncertainty_map.max()
        features.extend([uncertainty_mean, uncertainty_max])
        
        # 4. 元信息
        step_ratio = self.step_count / self.max_steps
        num_candidates_ratio = min(1.0, len(self.candidates) / self.max_select_candidates)
        num_selected_ratio = len(self.selected_indices) / max(1, len(self.candidates))
        features.extend([step_ratio, num_candidates_ratio, num_selected_ratio])
        
        # 填充到 obs_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.obs_dim:
            features = np.pad(features, (0, self.obs_dim - len(features)), mode='constant')
        elif len(features) > self.obs_dim:
            features = features[:self.obs_dim]
        
        return features
    
    def _get_grayscale_image(self) -> np.ndarray:
        """获取灰度图像（用于 vesselness 计算）"""
        if len(self.current_image.shape) == 3:
            return cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
        else:
            return self.current_image
    
    def render(self, mode='rgb_array'):
        """渲染环境（可视化）"""
        if mode != 'rgb_array':
            raise NotImplementedError(f"渲染模式 {mode} 未实现")
        
        # 创建可视化图像
        vis = self.current_image.copy()
        
        # 叠加当前掩膜（蓝色）
        if self.current_mask.sum() > 0:
            vis[self.current_mask] = vis[self.current_mask] * 0.5 + np.array([0, 0, 255], dtype=np.uint8) * 0.5
        
        # 叠加 GT（绿色边界，如果有）
        if self.current_gt_mask is not None:
            contours, _ = cv2.findContours(
                self.current_gt_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        return vis
    
    def close(self):
        """关闭环境"""
        pass

