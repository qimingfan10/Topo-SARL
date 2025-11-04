"""
阶段B：Prompt Decision Env
RL智能体学习生成精确的SAM2提示点
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, List, Optional
import cv2

from rewards.reward_functions import compute_iou, RewardCalculator


class PromptDecisionEnv(gym.Env):
    """
    Prompt Decision环境：RL学习生成SAM2提示点
    
    动作空间：MultiDiscrete [3, 32, 32]
        - action_type (3): 0=add_positive, 1=add_negative, 2=terminate
        - grid_x (32): x坐标（0-31）
        - grid_y (32): y坐标（0-31）
    
    观察空间：Dict
        - image: 当前图像 (512, 512, 3)
        - mask: 当前掩膜 (512, 512)
        - points_info: 提示点信息 (60,) - 最多20个点×3特征
        - iou: 当前IoU (1,)
        - step: 当前步数 (1,)
    """
    
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(
        self,
        sam_generator,
        reward_calculator: RewardCalculator,
        config: dict
    ):
        """
        Args:
            sam_generator: SAM2候选生成器
            reward_calculator: 奖励计算器
            config: 配置字典
        """
        super().__init__()
        
        self.sam_generator = sam_generator
        self.reward_calculator = reward_calculator
        self.config = config
        
        # 环境配置
        env_config = config.get('env', {})
        self.max_steps = env_config.get('max_steps', 20)
        self.grid_size = env_config.get('grid_size', 32)
        self.image_size = tuple(env_config.get('image_size', [512, 512]))
        self.max_points = env_config.get('max_points', 20)
        
        # 奖励配置
        reward_config = config.get('reward', {})
        self.delta_iou_weight = reward_config.get('delta_iou_weight', 10.0)
        self.final_iou_weight = reward_config.get('final_iou_weight', 5.0)
        self.action_cost = reward_config.get('action_cost', -0.01)
        self.iou_decrease_penalty = reward_config.get('iou_decrease_penalty', -0.5)
        
        # 新增：最小步数和探索奖励
        self.min_steps = reward_config.get('min_steps', 5)
        self.min_steps_bonus = reward_config.get('min_steps_bonus', 0.2)
        self.exploration_bonus = reward_config.get('exploration_bonus', 0.05)
        self.bonus_scale = reward_config.get('bonus_scale', 0.3)  # Bonus缩放系数
        
        # 调试模式
        self.debug_mode = reward_config.get('debug_mode', False)
        
        # 网格配置
        self.grid_cell_size = self.image_size[0] // self.grid_size  # 16像素/格
        
        # 动作空间：[action_type, grid_x, grid_y]
        # 为了兼容Stable-Baselines3，使用单个Discrete并手动编解码
        # action_type: 0=positive, 1=negative, 2=terminate
        # total = 3 * 32 * 32 = 3072
        self.action_space = spaces.Discrete(3 * self.grid_size * self.grid_size)
        
        # 观察空间：简化版（先用flatten）
        # 后续可以优化为CNN特征
        image_dim = self.image_size[0] * self.image_size[1] * 3
        mask_dim = self.image_size[0] * self.image_size[1]
        points_dim = self.max_points * 3  # (x, y, label) × 20
        scalar_dim = 2  # iou + step
        
        obs_dim = image_dim + mask_dim + points_dim + scalar_dim
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 状态变量
        self.current_image = None
        self.current_gt_mask = None
        self.current_mask = None
        self.current_points = []  # List of (x, y, label)
        self.current_iou = 0.0
        self.prev_iou = 0.0
        self.step_count = 0
        self.done = False
        
        print(f"✓ PromptDecisionEnv 初始化完成")
        print(f"  - 动作空间: Discrete({self.action_space.n})")
        print(f"  - 观察空间: Box({obs_dim},)")
        print(f"  - 最大步数: {self.max_steps}")
        print(f"  - 网格大小: {self.grid_size}×{self.grid_size}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 选项，必须包含 'image' 和 'gt_mask'
        
        Returns:
            observation: 初始观察
            info: 额外信息
        """
        super().reset(seed=seed)
        
        if options is None or 'image' not in options:
            raise ValueError("reset() 需要 options 参数，必须包含 'image'")
        
        # 获取图像和标注
        self.current_image = options['image']
        self.current_gt_mask = options.get('gt_mask', None)
        
        # 初始化状态
        self.current_mask = np.zeros(self.image_size, dtype=bool)
        self.current_points = []
        self.current_iou = 0.0
        self.prev_iou = 0.0
        self.step_count = 0
        self.done = False
        
        # 设置SAM2的图像（预计算特征）
        self.sam_generator.set_image(self.current_image)
        
        # 返回观察
        obs = self._get_observation()
        info = {
            'current_iou': self.current_iou,
            'step_count': self.step_count,
            'num_points': len(self.current_points)
        }
        
        return obs, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 编码后的动作（0-3071）
        
        Returns:
            observation: 新观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        if self.done:
            raise RuntimeError("环境已终止，请调用 reset()")
        
        self.step_count += 1
        
        # 解码动作
        action_type, grid_x, grid_y = self._decode_action(action)
        
        terminated = False
        truncated = False
        reward = 0.0
        info = {
            'action_type': action_type,
            'action_name': ['positive', 'negative', 'terminate'][action_type],
            'grid_pos': (grid_x, grid_y)
        }
        
        # 执行动作
        if action_type == 2:  # Terminate
            terminated = True
            info['action_detail'] = 'terminate'
        else:
            # 转换网格坐标到像素坐标（网格中心）
            pixel_x = grid_x * self.grid_cell_size + self.grid_cell_size // 2
            pixel_y = grid_y * self.grid_cell_size + self.grid_cell_size // 2
            
            # 确保坐标在图像范围内
            pixel_x = np.clip(pixel_x, 0, self.image_size[1] - 1)
            pixel_y = np.clip(pixel_y, 0, self.image_size[0] - 1)
            
            # 添加提示点
            point_label = 1 if action_type == 0 else 0  # 1=positive, 0=negative
            self.current_points.append((pixel_x, pixel_y, point_label))
            
            # 限制提示点数量
            if len(self.current_points) > self.max_points:
                self.current_points.pop(0)  # 移除最早的点
            
            # 调用SAM2生成新掩膜
            self._update_mask_with_sam2()
            
            info['action_detail'] = f"{'positive' if point_label == 1 else 'negative'}_at_({pixel_x},{pixel_y})"
            info['num_points'] = len(self.current_points)
        
        # 计算当前IoU
        if self.current_gt_mask is not None:
            self.current_iou = compute_iou(self.current_mask, self.current_gt_mask)
            info['current_iou'] = self.current_iou
        
        # 计算奖励
        reward = self._compute_reward(action_type)
        info['reward'] = reward
        info['reward_components'] = self._get_reward_components(action_type)
        
        # 更新prev_iou
        self.prev_iou = self.current_iou
        
        # 检查是否超时
        if self.step_count >= self.max_steps:
            truncated = True
        
        self.done = terminated or truncated
        
        # 记录最终IoU
        if self.done:
            info['final_iou'] = self.current_iou
            info['final_area'] = int(self.current_mask.sum())
        
        # 获取新观察
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _decode_action(self, flat_action: int) -> Tuple[int, int, int]:
        """
        解码扁平动作到 (action_type, grid_x, grid_y)
        
        Args:
            flat_action: 0-3071
        
        Returns:
            action_type: 0=positive, 1=negative, 2=terminate
            grid_x: 0-31
            grid_y: 0-31
        """
        action_type = flat_action // (self.grid_size * self.grid_size)
        remainder = flat_action % (self.grid_size * self.grid_size)
        grid_x = remainder // self.grid_size
        grid_y = remainder % self.grid_size
        
        return action_type, grid_x, grid_y
    
    def _update_mask_with_sam2(self):
        """使用当前提示点调用SAM2生成新掩膜"""
        if len(self.current_points) == 0:
            self.current_mask = np.zeros(self.image_size, dtype=bool)
            return
        
        # 准备SAM2输入
        points = np.array([(x, y) for x, y, _ in self.current_points], dtype=np.float32)
        labels = np.array([label for _, _, label in self.current_points], dtype=np.int32)
        
        # 调用SAM2
        masks, scores, _ = self.sam_generator.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False  # 只要一个掩膜
        )
        
        # 提取掩膜（第一个）
        if masks is not None and len(masks) > 0:
            self.current_mask = masks[0].astype(bool)
        else:
            self.current_mask = np.zeros(self.image_size, dtype=bool)
    
    def _compute_reward(self, action_type: int) -> float:
        """
        计算奖励（优化版本）
        
        Args:
            action_type: 动作类型
        
        Returns:
            reward: 总奖励
        """
        reward = 0.0
        delta_iou = self.current_iou - self.prev_iou
        
        # 1. 增量IoU奖励（主要奖励信号）
        reward += delta_iou * self.delta_iou_weight
        
        # 2. 动作成本
        reward += self.action_cost
        
        # 3. IoU下降惩罚
        if delta_iou < -0.01:
            reward += self.iou_decrease_penalty
        
        # 4. 最小步数奖励（鼓励探索）
        if self.step_count >= self.min_steps:
            reward += self.min_steps_bonus
        
        # 5. 探索奖励（超过最小步数后的额外奖励）
        if self.step_count > self.min_steps:
            reward += self.exploration_bonus
        
        # 6. 最终奖励（terminate时）
        if action_type == 2:
            reward += self.current_iou * self.final_iou_weight
            
            # 优化的Bonus（减小权重）
            if self.current_iou > 0.5:
                reward += 0.5 * self.bonus_scale  # 从2.0减到0.15
            elif self.current_iou > 0.3:
                reward += 0.3 * self.bonus_scale  # 从1.0减到0.09
            elif self.current_iou > 0.1:
                reward += 0.1 * self.bonus_scale  # 从0.5减到0.03
            
            # 过早终止惩罚
            if self.step_count < self.min_steps:
                reward -= 0.5  # 惩罚过早终止
        
        # 调试信息
        if self.debug_mode and self.step_count % 5 == 0:
            print(f"  [DEBUG] Step={self.step_count}, IoU={self.current_iou:.4f}, "
                  f"delta_IoU={delta_iou:.4f}, reward={reward:.4f}")
        
        return reward
    
    def _get_reward_components(self, action_type: int) -> Dict[str, float]:
        """获取奖励各组成部分（用于调试）"""
        delta_iou = self.current_iou - self.prev_iou
        
        components = {
            'delta_iou': delta_iou,
            'delta_iou_reward': delta_iou * self.delta_iou_weight,
            'action_cost': self.action_cost,
            'iou_decrease_penalty': self.iou_decrease_penalty if delta_iou < -0.01 else 0.0,
            'min_steps_bonus': self.min_steps_bonus if self.step_count >= self.min_steps else 0.0,
            'exploration_bonus': self.exploration_bonus if self.step_count > self.min_steps else 0.0,
        }
        
        if action_type == 2:
            components['final_iou_reward'] = self.current_iou * self.final_iou_weight
            if self.current_iou > 0.5:
                components['bonus'] = 0.5 * self.bonus_scale
            elif self.current_iou > 0.3:
                components['bonus'] = 0.3 * self.bonus_scale
            elif self.current_iou > 0.1:
                components['bonus'] = 0.1 * self.bonus_scale
            else:
                components['bonus'] = 0.0
            
            # 过早终止惩罚
            if self.step_count < self.min_steps:
                components['early_terminate_penalty'] = -0.5
        
        return components
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察
        
        Returns:
            obs: 扁平化的观察向量
        """
        # 归一化图像（0-255 -> 0-1）
        image_norm = self.current_image.astype(np.float32) / 255.0
        image_flat = image_norm.flatten()
        
        # 掩膜（bool -> float）
        mask_flat = self.current_mask.astype(np.float32).flatten()
        
        # 提示点信息（填充到max_points）
        points_array = np.zeros((self.max_points, 3), dtype=np.float32)
        for i, (x, y, label) in enumerate(self.current_points[:self.max_points]):
            points_array[i] = [
                x / self.image_size[1],  # 归一化x
                y / self.image_size[0],  # 归一化y
                float(label)
            ]
        points_flat = points_array.flatten()
        
        # 标量特征
        iou_scalar = np.array([self.current_iou], dtype=np.float32)
        step_scalar = np.array([self.step_count / self.max_steps], dtype=np.float32)
        
        # 拼接所有特征
        obs = np.concatenate([
            image_flat,
            mask_flat,
            points_flat,
            iou_scalar,
            step_scalar
        ])
        
        return obs
    
    def render(self, mode='rgb_array'):
        """渲染环境（可选）"""
        if mode == 'rgb_array':
            # 创建可视化图像
            vis = self.current_image.copy()
            
            # 叠加掩膜（半透明绿色）
            if self.current_mask.sum() > 0:
                mask_overlay = np.zeros_like(vis)
                mask_overlay[self.current_mask] = [0, 255, 0]
                vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
            
            # 绘制提示点
            for x, y, label in self.current_points:
                color = (0, 255, 0) if label == 1 else (255, 0, 0)  # 绿=正，红=负
                cv2.circle(vis, (int(x), int(y)), 5, color, -1)
                cv2.circle(vis, (int(x), int(y)), 7, (255, 255, 255), 1)
            
            # 添加文字信息
            info_text = f"Step: {self.step_count}/{self.max_steps} | IoU: {self.current_iou:.4f} | Points: {len(self.current_points)}"
            cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return vis
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")
    
    def close(self):
        """关闭环境"""
        pass

