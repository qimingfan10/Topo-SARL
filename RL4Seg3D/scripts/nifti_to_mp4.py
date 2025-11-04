#!/usr/bin/env python3
"""
将NIfTI格式的.nii.gz文件转换为MP4视频
支持单个文件或批量转换整个目录
"""

import argparse
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm


def normalize_to_uint8(data):
    """将数据归一化到0-255范围的uint8格式"""
    if data.dtype == np.bool_ or data.dtype == bool:
        return (data.astype(np.uint8) * 255)
    
    data = data.astype(np.float32)
    data_min = data.min()
    data_max = data.max()
    
    if data_max - data_min > 0:
        data = (data - data_min) / (data_max - data_min) * 255
    else:
        data = np.zeros_like(data)
    
    return data.astype(np.uint8)


def create_overlay(image, mask, alpha=0.5):
    """创建图像和分割mask的叠加视图"""
    # 确保都是uint8格式
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 为mask创建彩色版本
    mask_colored = np.zeros_like(image)
    if mask.max() > 0:
        # 不同的标签用不同颜色
        unique_labels = np.unique(mask[mask > 0])
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
        ]
        for idx, label in enumerate(unique_labels):
            color = colors[idx % len(colors)]
            mask_colored[mask == label] = color
    
    # 叠加
    overlay = cv2.addWeighted(image, 1-alpha, mask_colored, alpha, 0)
    return overlay


def nifti_to_mp4(nifti_path, output_path, fps=5, axis=2, overlay_mask=None, 
                 resize_width=None, codec='mp4v'):
    """
    将NIfTI文件转换为MP4视频
    
    参数:
        nifti_path: 输入的.nii.gz文件路径
        output_path: 输出的.mp4文件路径
        fps: 视频帧率，默认10
        axis: 沿哪个轴切片 (0, 1, 或 2)，默认2（通常是深度/时间轴）
        overlay_mask: 可选的mask文件路径，用于叠加显示
        resize_width: 可选的输出宽度，用于缩放视频
        codec: 视频编码器，默认'mp4v'
    """
    print(f"正在读取: {nifti_path}")
    
    # 加载NIfTI文件
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    
    # 加载mask（如果提供）
    mask_data = None
    if overlay_mask and Path(overlay_mask).exists():
        print(f"正在加载mask: {overlay_mask}")
        mask_nii = nib.load(overlay_mask)
        mask_data = mask_nii.get_fdata()
    
    print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
    
    # 归一化数据
    data_uint8 = normalize_to_uint8(data)
    if mask_data is not None:
        mask_uint8 = mask_data.astype(np.uint8)
    
    # 根据指定轴重新排列数据
    if axis != 2:
        data_uint8 = np.moveaxis(data_uint8, axis, 2)
        if mask_data is not None:
            mask_uint8 = np.moveaxis(mask_uint8, axis, 2)
    
    # 获取帧数和尺寸
    height, width, num_frames = data_uint8.shape
    
    # 调整尺寸（如果指定）
    if resize_width:
        resize_height = int(height * resize_width / width)
        height, width = resize_height, resize_width
    
    print(f"输出视频: {width}x{height}, {num_frames}帧, {fps}fps")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"警告: 无法使用编码器 '{codec}'，尝试使用 'avc1'")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")
    
    # 逐帧写入
    for i in tqdm(range(num_frames), desc="转换进度"):
        frame = data_uint8[:, :, i]
        
        # 转换为BGR（OpenCV格式）
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
        
        # 如果有mask，创建叠加
        if mask_data is not None:
            mask_frame = mask_uint8[:, :, i]
            frame_bgr = create_overlay(frame_bgr, mask_frame, alpha=0.4)
        
        # 调整尺寸
        if resize_width:
            frame_bgr = cv2.resize(frame_bgr, (width, height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        out.write(frame_bgr)
    
    out.release()
    print(f"✓ 转换完成: {output_path}")


def batch_convert_directory(input_dir, output_dir, pattern="*.nii.gz", 
                           fps=5, axis=2, overlay_pred=False):
    """
    批量转换目录中的所有NIfTI文件
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        pattern: 文件匹配模式，默认"*.nii.gz"
        fps: 视频帧率
        axis: 切片轴
        overlay_pred: 是否叠加预测结果
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有匹配的文件
    nifti_files = list(input_path.glob(pattern))
    
    if not nifti_files:
        print(f"在 {input_dir} 中没有找到匹配 '{pattern}' 的文件")
        return
    
    print(f"找到 {len(nifti_files)} 个文件")
    
    for nifti_file in nifti_files:
        # 生成输出文件名
        output_file = output_path / f"{nifti_file.stem.replace('.nii', '')}.mp4"
        
        # 查找对应的预测mask（如果需要叠加）
        mask_file = None
        if overlay_pred:
            # 尝试在pred目录中查找对应的文件
            pred_dir = input_path.parent / "pred"
            if pred_dir.exists():
                mask_file = pred_dir / nifti_file.name
                if not mask_file.exists():
                    mask_file = None
        
        try:
            nifti_to_mp4(
                nifti_file, 
                output_file, 
                fps=fps, 
                axis=axis,
                overlay_mask=mask_file
            )
        except Exception as e:
            print(f"✗ 转换失败 {nifti_file.name}: {e}")


def convert_reward_dataset(reward_ds_dir, output_dir, fps=5):
    """
    转换RewardDataset格式的目录（包含images/gt/pred三个子目录）
    
    参数:
        reward_ds_dir: RewardDataset根目录
        output_dir: 输出目录
        fps: 视频帧率
    """
    reward_path = Path(reward_ds_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = reward_path / "images"
    gt_dir = reward_path / "gt"
    pred_dir = reward_path / "pred"
    
    if not images_dir.exists():
        print(f"错误: 找不到images目录: {images_dir}")
        return
    
    image_files = list(images_dir.glob("*.nii.gz"))
    print(f"找到 {len(image_files)} 个图像文件")
    
    for img_file in image_files:
        base_name = img_file.stem.replace('.nii', '')
        
        # 为每个文件创建三个视频
        print(f"\n处理: {base_name}")
        
        # 1. 原始图像
        try:
            output_file = output_path / f"{base_name}_image.mp4"
            nifti_to_mp4(img_file, output_file, fps=fps)
        except Exception as e:
            print(f"  ✗ 图像转换失败: {e}")
        
        # 2. 图像+GT叠加
        gt_file = gt_dir / img_file.name
        if gt_file.exists():
            try:
                output_file = output_path / f"{base_name}_with_gt.mp4"
                nifti_to_mp4(img_file, output_file, fps=fps, overlay_mask=gt_file)
            except Exception as e:
                print(f"  ✗ GT叠加转换失败: {e}")
        
        # 3. 图像+预测叠加
        pred_file = pred_dir / img_file.name
        if pred_file.exists():
            try:
                output_file = output_path / f"{base_name}_with_pred.mp4"
                nifti_to_mp4(img_file, output_file, fps=fps, overlay_mask=pred_file)
            except Exception as e:
                print(f"  ✗ 预测叠加转换失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="将NIfTI格式(.nii.gz)文件转换为MP4视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  # 转换单个文件
  python nifti_to_mp4.py -i input.nii.gz -o output.mp4

  # 批量转换目录
  python nifti_to_mp4.py -i /path/to/nifti/dir -o /path/to/output/dir --batch

  # 转换RewardDataset目录（自动处理images/gt/pred）
  python nifti_to_mp4.py -i /path/to/rewardDS -o /path/to/videos --reward-dataset

  # 自定义帧率和轴
  python nifti_to_mp4.py -i input.nii.gz -o output.mp4 --fps 15 --axis 0
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入文件或目录路径')
    parser.add_argument('-o', '--output', required=True,
                       help='输出文件或目录路径')
    parser.add_argument('--fps', type=int, default=5,
                       help='视频帧率 (默认: 5fps，对于少量帧的数据)')
    parser.add_argument('--axis', type=int, default=2, choices=[0, 1, 2],
                       help='切片轴 (0, 1, 或 2, 默认: 2)')
    parser.add_argument('--batch', action='store_true',
                       help='批量转换目录中的所有文件')
    parser.add_argument('--reward-dataset', action='store_true',
                       help='转换RewardDataset格式的目录')
    parser.add_argument('--pattern', default='*.nii.gz',
                       help='批量模式下的文件匹配模式 (默认: *.nii.gz)')
    parser.add_argument('--overlay-pred', action='store_true',
                       help='批量模式下叠加预测结果')
    parser.add_argument('--width', type=int,
                       help='输出视频宽度（保持宽高比）')
    
    args = parser.parse_args()
    
    if args.reward_dataset:
        # RewardDataset模式
        convert_reward_dataset(args.input, args.output, fps=args.fps)
    elif args.batch:
        # 批量转换模式
        batch_convert_directory(
            args.input, 
            args.output, 
            pattern=args.pattern,
            fps=args.fps,
            axis=args.axis,
            overlay_pred=args.overlay_pred
        )
    else:
        # 单文件转换模式
        nifti_to_mp4(
            args.input, 
            args.output, 
            fps=args.fps,
            axis=args.axis,
            resize_width=args.width
        )


if __name__ == '__main__':
    main()

