"""
像素级分类环境 - 任务重定向
不使用SAM2生成掩膜，而是让RL agent直接进行像素级分类
"""
import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Optional
from gymnasium import spaces


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算IoU"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)


class PixelClassificationEnv:
    """
    像素级分类环境
    
    Agent在grid上逐个分类像素块：
    - Action: 选择一个grid cell + 预测类别（血管/背景）
    - Observation: 图像 + 当前分类状态
    - Reward: 基于分类准确率和IoU
    """
    
    def __init__(self, config: Dict):
        # 环境配置
        self.max_steps = config.get('max_steps', 50)
        self.grid_size = config.get('grid_size', 16)  # 16x16 grid
        self.image_size = tuple(config.get('image_size', [512, 512]))
        
        # 计算grid cell大小
        self.cell_height = self.image_size[0] // self.grid_size
        self.cell_width = self.image_size[1] // self.grid_size
        
        # 奖励配置
        reward_config = config.get('reward', {})
        self.iou_weight = reward_config.get('iou_weight', 100.0)
        self.precision_weight = reward_config.get('precision_weight', 50.0)
        self.recall_weight = reward_config.get('recall_weight', 50.0)
        self.action_cost = reward_config.get('action_cost', -0.1)
        self.correct_classification_bonus = reward_config.get('correct_classification_bonus', 1.0)
        self.wrong_classification_penalty = reward_config.get('wrong_classification_penalty', -0.5)
        
        # 策略配置
        self.auto_classify_strategy = reward_config.get('auto_classify_strategy', 'threshold')  # 'threshold' or 'none'
        self.auto_classify_threshold = reward_config.get('auto_classify_threshold', 0.5)
        
        # 调试配置
        self.debug_mode = config.get('debug_mode', True)
        self.debug_interval = config.get('debug_interval', 5)
        
        # Action space: [grid_x, grid_y, label]
        # label: 0=背景, 1=血管
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size * 2)
        
        # Observation space: [image, current_prediction_mask, classified_mask]
        # image: HxWx3
        # current_prediction_mask: HxW (0/1)
        # classified_mask: HxW (0=未分类, 1=已分类)
        obs_shape = (
            self.image_size[0] * self.image_size[1] * 3 +  # image
            self.image_size[0] * self.image_size[1] +      # prediction mask
            self.image_size[0] * self.image_size[1] +      # classified mask
            1  # step_count
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_shape,), dtype=np.float32
        )
        
        # 状态变量
        self.current_image = None
        self.current_gt_mask = None
        self.current_prediction = None  # HxW, 0=背景, 1=血管
        self.classified_grid = None     # grid_size x grid_size, bool, 是否已分类
        self.step_count = 0
        self.done = False
        
        # 统计
        self.total_pixels = self.image_size[0] * self.image_size[1]
        self.grid_total = self.grid_size * self.grid_size
        
        # 调试统计
        self.episode_stats = {
            'correct_classifications': 0,
            'wrong_classifications': 0,
            'vessel_cells_found': 0,
            'vessel_cells_total': 0,
            'iou_history': [],
            'precision_history': [],
            'recall_history': []
        }
    
    def reset(self, image: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
        """重置环境"""
        # 设置图像和GT
        self.current_image = cv2.resize(image, self.image_size[::-1])  # (W, H)
        # 确保mask是uint8类型再resize
        if gt_mask.dtype == bool:
            gt_mask = gt_mask.astype(np.uint8) * 255
        self.current_gt_mask = cv2.resize(gt_mask, self.image_size[::-1]) > 0
        
        # 归一化图像
        if self.current_image.dtype == np.uint8:
            self.current_image = self.current_image.astype(np.float32) / 255.0
        
        # 初始化预测mask（全为背景）
        self.current_prediction = np.zeros(self.image_size, dtype=bool)
        
        # 初始化已分类grid（全为False）
        self.classified_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # 重置状态
        self.step_count = 0
        self.done = False
        
        # 计算GT中的血管grid cells
        vessel_grid_count = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i * self.cell_height, (i + 1) * self.cell_height
                x1, x2 = j * self.cell_width, (j + 1) * self.cell_width
                cell_gt = self.current_gt_mask[y1:y2, x1:x2]
                if cell_gt.sum() > (cell_gt.size * 0.3):  # >30%为血管
                    vessel_grid_count += 1
        
        self.episode_stats = {
            'correct_classifications': 0,
            'wrong_classifications': 0,
            'vessel_cells_found': 0,
            'vessel_cells_total': vessel_grid_count,
            'iou_history': [],
            'precision_history': [],
            'recall_history': []
        }
        
        if self.debug_mode:
            print(f"\n[RESET] 新episode开始")
            print(f"  图像大小: {self.image_size}")
            print(f"  Grid大小: {self.grid_size}x{self.grid_size} = {self.grid_total}个cells")
            print(f"  Cell大小: {self.cell_height}x{self.cell_width}")
            print(f"  血管占比: {self.current_gt_mask.sum() / self.total_pixels * 100:.2f}%")
            print(f"  血管cells: {vessel_grid_count}/{self.grid_total} ({vessel_grid_count/self.grid_total*100:.1f}%)")
        
        return self._get_observation()
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """解码动作"""
        label = action % 2
        grid_idx = action // 2
        grid_y = grid_idx // self.grid_size
        grid_x = grid_idx % self.grid_size
        return grid_x, grid_y, label
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        if self.done:
            raise RuntimeError("环境已终止，请调用reset()")
        
        self.step_count += 1
        
        # 解码动作
        grid_x, grid_y, label = self._decode_action(action)
        
        # 检查是否已分类
        if self.classified_grid[grid_y, grid_x]:
            # 已分类过，给予小惩罚
            reward = -0.2
            info = {
                'action': f'duplicate_({grid_x},{grid_y})',
                'reward': reward,
                'already_classified': True
            }
            
            if self.debug_mode and self.step_count % self.debug_interval == 0:
                print(f"  [STEP {self.step_count}] 重复分类 ({grid_x},{grid_y}), R={reward:.2f}")
            
        else:
            # 标记为已分类
            self.classified_grid[grid_y, grid_x] = True
            
            # 获取cell范围
            y1, y2 = grid_y * self.cell_height, (grid_y + 1) * self.cell_height
            x1, x2 = grid_x * self.cell_width, (grid_x + 1) * self.cell_width
            
            # 获取cell的GT
            cell_gt = self.current_gt_mask[y1:y2, x1:x2]
            gt_vessel_ratio = cell_gt.sum() / cell_gt.size
            
            # 判断GT label（>30%为血管）
            gt_label = 1 if gt_vessel_ratio > 0.3 else 0
            
            # 更新预测mask
            if label == 1:
                self.current_prediction[y1:y2, x1:x2] = True
            # else: 保持为False（背景）
            
            # 计算奖励
            reward = self._compute_reward(grid_x, grid_y, label, gt_label, gt_vessel_ratio)
            
            # 统计
            if label == gt_label:
                self.episode_stats['correct_classifications'] += 1
            else:
                self.episode_stats['wrong_classifications'] += 1
            
            if label == 1 and gt_label == 1:
                self.episode_stats['vessel_cells_found'] += 1
            
            info = {
                'action': f'{["bg", "vessel"][label]}_({grid_x},{grid_y})',
                'gt_label': gt_label,
                'gt_vessel_ratio': gt_vessel_ratio,
                'correct': label == gt_label,
                'reward': reward,
                'already_classified': False
            }
            
            # 调试输出
            if self.debug_mode and (self.step_count % self.debug_interval == 0 or label == gt_label):
                correct_mark = "✓" if label == gt_label else "✗"
                print(f"  [STEP {self.step_count}] ({grid_x},{grid_y}) pred={label} gt={gt_label} {correct_mark} "
                      f"gt_ratio={gt_vessel_ratio*100:.1f}% R={reward:+.2f}")
        
        # 检查是否所有cells都分类完成
        classified_count = self.classified_grid.sum()
        terminated = (classified_count == self.grid_total) or (self.step_count >= self.max_steps)
        
        if terminated:
            self.done = True
            # 最终奖励
            final_reward = self._compute_final_reward()
            reward += final_reward
            info['final_reward'] = final_reward
            info['classified_ratio'] = classified_count / self.grid_total
            
            if self.debug_mode:
                self._print_final_stats()
        
        return self._get_observation(), reward, terminated, False, info
    
    def _compute_reward(self, grid_x: int, grid_y: int, pred_label: int, 
                       gt_label: int, gt_vessel_ratio: float) -> float:
        """计算单步奖励"""
        reward = 0.0
        
        # 1. 动作成本
        reward += self.action_cost
        
        # 2. 分类正确性奖励
        if pred_label == gt_label:
            reward += self.correct_classification_bonus
            # 额外奖励：如果是血管cell且预测正确
            if gt_label == 1:
                reward += self.correct_classification_bonus * 2.0  # 血管更重要
        else:
            reward += self.wrong_classification_penalty
            # 额外惩罚：false positive或false negative
            if pred_label == 1 and gt_label == 0:
                # False positive
                reward += self.wrong_classification_penalty * 0.5
            elif pred_label == 0 and gt_label == 1:
                # False negative（更严重）
                reward += self.wrong_classification_penalty * 1.5
        
        # 3. 基于gt_vessel_ratio的细粒度奖励
        if gt_vessel_ratio > 0.5 and pred_label == 1:
            # 高置信度血管，预测正确
            reward += 0.5
        elif gt_vessel_ratio < 0.1 and pred_label == 0:
            # 高置信度背景，预测正确
            reward += 0.3
        
        return reward
    
    def _compute_final_reward(self) -> float:
        """计算最终奖励"""
        # 计算性能指标
        iou = compute_iou(self.current_prediction, self.current_gt_mask)
        
        # 计算precision和recall
        tp = np.logical_and(self.current_prediction, self.current_gt_mask).sum()
        fp = np.logical_and(self.current_prediction, ~self.current_gt_mask).sum()
        fn = np.logical_and(~self.current_prediction, self.current_gt_mask).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # 保存历史
        self.episode_stats['iou_history'].append(iou)
        self.episode_stats['precision_history'].append(precision)
        self.episode_stats['recall_history'].append(recall)
        
        # 计算奖励
        reward = (
            iou * self.iou_weight +
            precision * self.precision_weight +
            recall * self.recall_weight
        )
        
        # 里程碑奖励
        if iou >= 0.10:
            reward += 50.0
        elif iou >= 0.08:
            reward += 30.0
        elif iou >= 0.05:
            reward += 15.0
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        # 展平所有信息
        image_flat = self.current_image.flatten()
        prediction_flat = self.current_prediction.astype(np.float32).flatten()
        classified_flat = np.repeat(
            np.repeat(self.classified_grid, self.cell_height, axis=0),
            self.cell_width, axis=1
        ).astype(np.float32).flatten()
        step_norm = np.array([self.step_count / self.max_steps], dtype=np.float32)
        
        obs = np.concatenate([image_flat, prediction_flat, classified_flat, step_norm])
        return obs
    
    def _print_final_stats(self):
        """打印最终统计"""
        iou = self.episode_stats['iou_history'][-1] if self.episode_stats['iou_history'] else 0
        precision = self.episode_stats['precision_history'][-1] if self.episode_stats['precision_history'] else 0
        recall = self.episode_stats['recall_history'][-1] if self.episode_stats['recall_history'] else 0
        
        correct = self.episode_stats['correct_classifications']
        wrong = self.episode_stats['wrong_classifications']
        total = correct + wrong
        accuracy = correct / total * 100 if total > 0 else 0
        
        vessel_found = self.episode_stats['vessel_cells_found']
        vessel_total = self.episode_stats['vessel_cells_total']
        vessel_recall = vessel_found / vessel_total * 100 if vessel_total > 0 else 0
        
        print(f"\n[EPISODE END] Step {self.step_count}")
        print(f"  分类准确率: {correct}/{total} = {accuracy:.1f}%")
        print(f"  血管召回: {vessel_found}/{vessel_total} = {vessel_recall:.1f}%")
        print(f"  IoU: {iou*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall: {recall*100:.2f}%")
        print(f"  F1-Score: {2*precision*recall/(precision+recall+1e-8)*100:.2f}%")

