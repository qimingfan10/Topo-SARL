#!/usr/bin/env python3
"""
将Segment_DATA数据集转换为RL4Seg3D需要的格式

输入格式：
- orgin_pic/: 原始图像（JPG）
- lab_pic/: 标注图像（PNG）
- json/: 标注JSON

输出格式：
- my_organized_dataset/
  ├── img/
  │   └── {study}/
  │       └── {view}/
  │           └── {dicom_uuid}_0000.nii.gz
  ├── segmentation/
  │   └── {study}/
  │       └── {view}/
  │           └── {dicom_uuid}.nii.gz
  └── my_organized_dataset.csv
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
from tqdm import tqdm

def parse_filename(filename):
    """
    解析文件名：患者名(ID)_视图_UUID_frame_帧号.jpg
    返回：(patient_name, patient_id, view, uuid, frame_num)
    """
    # 移除扩展名
    name = filename.replace('.jpg', '').replace('.png', '').replace('_mask', '')
    
    # 分割文件名
    parts = name.split('_')
    
    # 患者名(ID)
    patient_info = parts[0]  # "An Cong Xue(0000932433)"
    patient_name = patient_info.split('(')[0]
    patient_id = patient_info.split('(')[1].replace(')', '')
    
    # 视图
    view = parts[1]  # "1-3"
    
    # 其他部分
    rest_part = parts[2]  # "1"
    uuid = parts[3]  # "051C3E6A"
    
    # 帧号
    frame = parts[-1].replace('frame_', '')  # "000011"
    
    return patient_name, patient_id, view, uuid, int(frame)

def group_frames_by_sequence(image_dir):
    """
    按序列（UUID）分组帧
    返回：{sequence_key: [frame_files]}
    """
    sequences = defaultdict(list)
    
    for img_file in os.listdir(image_dir):
        if not img_file.endswith('.jpg'):
            continue
        
        patient_name, patient_id, view, uuid, frame_num = parse_filename(img_file)
        
        # 序列key: PatientID_View_UUID
        seq_key = f"{patient_id}_{view}_{uuid}"
        
        sequences[seq_key].append({
            'filename': img_file,
            'patient_name': patient_name,
            'patient_id': patient_id,
            'view': view,
            'uuid': uuid,
            'frame_num': frame_num
        })
    
    # 按帧号排序
    for seq_key in sequences:
        sequences[seq_key].sort(key=lambda x: x['frame_num'])
    
    return sequences

def load_frames(image_dir, mask_dir, frames_info):
    """
    加载一个序列的所有帧（图像和mask）
    返回：(images_array, masks_array) 形状 [H, W, num_frames]
    """
    images = []
    masks = []
    
    # 先加载第一帧确定尺寸
    first_img_path = Path(image_dir) / frames_info[0]['filename']
    first_img = Image.open(first_img_path).convert('L')
    target_size = first_img.size  # (W, H)
    
    for frame_info in frames_info:
        # 加载原始图像
        img_path = Path(image_dir) / frame_info['filename']
        img = Image.open(img_path).convert('L')  # 转灰度
        
        # 确保尺寸一致
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)
        
        images.append(np.array(img))
        
        # 加载mask
        mask_filename = frame_info['filename'].replace('.jpg', '_mask.png')
        mask_path = Path(mask_dir) / mask_filename
        
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            
            # 调整mask到与图像相同的尺寸
            if mask.size != target_size:
                mask = mask.resize(target_size, Image.NEAREST)  # 用最近邻保持标签
            
            mask_array = np.array(mask)
            # 二值化mask（假设mask是0或255，或者是灰度值）
            mask_array = (mask_array > 127).astype(np.uint8)
        else:
            # 如果没有mask，创建空mask
            mask_array = np.zeros_like(images[-1], dtype=np.uint8)
        
        masks.append(mask_array)
    
    # 堆叠成3D数组 [H, W, D]
    images_3d = np.stack(images, axis=-1)
    masks_3d = np.stack(masks, axis=-1)
    
    return images_3d, masks_3d

def save_as_nifti(data, output_path, is_mask=False):
    """
    保存为NIfTI格式
    """
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建NIfTI图像
    if is_mask:
        # Mask使用整数类型
        nifti_img = nib.Nifti1Image(data.astype(np.int16), affine=np.eye(4))
    else:
        # 图像归一化到[0, 1]
        data_normalized = data.astype(np.float32) / 255.0
        nifti_img = nib.Nifti1Image(data_normalized, affine=np.eye(4))
    
    # 保存
    nib.save(nifti_img, str(output_path))

def convert_dataset(source_dir, target_dir):
    """
    转换整个数据集
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建输出目录
    img_dir = target_path / 'img'
    seg_dir = target_path / 'segmentation'
    
    # 输入目录
    orgin_pic_dir = source_path / 'orgin_pic'
    lab_pic_dir = source_path / 'lab_pic'
    
    print(f"源目录: {source_path}")
    print(f"目标目录: {target_path}")
    print(f"\n正在分析数据集...")
    
    # 分组序列
    sequences = group_frames_by_sequence(str(orgin_pic_dir))
    
    print(f"找到 {len(sequences)} 个序列")
    
    # CSV数据
    csv_data = []
    
    # 转换每个序列
    for seq_key, frames_info in tqdm(sequences.items(), desc="转换序列"):
        # 获取序列信息
        first_frame = frames_info[0]
        patient_id = first_frame['patient_id']
        patient_name = first_frame['patient_name']
        view = first_frame['view']
        uuid = first_frame['uuid']
        
        # 创建study名称（使用患者名）
        study_name = patient_name.replace(' ', '_')
        view_name = f"view_{view}"
        
        # 加载帧数据
        try:
            images_3d, masks_3d = load_frames(orgin_pic_dir, lab_pic_dir, frames_info)
        except Exception as e:
            print(f"\n警告：跳过序列 {seq_key}: {e}")
            continue
        
        # 保存图像
        img_output_dir = img_dir / study_name / view_name
        img_filename = f"{uuid}_0000.nii.gz"
        img_output_path = img_output_dir / img_filename
        
        save_as_nifti(images_3d, img_output_path, is_mask=False)
        
        # 保存mask
        seg_output_dir = seg_dir / study_name / view_name
        seg_filename = f"{uuid}.nii.gz"
        seg_output_path = seg_output_dir / seg_filename
        
        save_as_nifti(masks_3d, seg_output_path, is_mask=True)
        
        # 添加到CSV
        csv_data.append({
            'study': study_name,
            'view': view_name,
            'dicom_uuid': uuid,
            'patient_id': patient_id,
            'patient_name': patient_name,
            'num_frames': len(frames_info),
            'H': images_3d.shape[0],
            'W': images_3d.shape[1],
            'D': images_3d.shape[2],
        })
    
    # 创建CSV文件
    print(f"\n创建CSV索引文件...")
    df = pd.DataFrame(csv_data)
    
    # 添加split列（80% train, 10% val, 10% test）
    n_total = len(df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    df['my_split'] = 'test'
    df.iloc[:n_train, df.columns.get_loc('my_split')] = 'train'
    df.iloc[n_train:n_train+n_val, df.columns.get_loc('my_split')] = 'val'
    
    # 保存CSV
    csv_path = target_path / 'my_organized_dataset.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ 转换完成！")
    print(f"  - 总序列数: {len(df)}")
    print(f"  - Train: {sum(df['my_split'] == 'train')}")
    print(f"  - Val: {sum(df['my_split'] == 'val')}")
    print(f"  - Test: {sum(df['my_split'] == 'test')}")
    print(f"  - 输出目录: {target_path}")
    print(f"  - CSV文件: {csv_path}")
    
    return df

def verify_conversion(target_dir):
    """
    验证转换结果
    """
    target_path = Path(target_dir)
    csv_path = target_path / 'my_organized_dataset.csv'
    
    print(f"\n验证转换结果...")
    
    if not csv_path.exists():
        print(f"✗ CSV文件不存在: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"✓ CSV文件包含 {len(df)} 条记录")
    
    # 验证文件存在
    missing_count = 0
    for _, row in df.iterrows():
        # 检查图像文件
        img_path = target_path / 'img' / row['study'] / row['view'] / f"{row['dicom_uuid']}_0000.nii.gz"
        if not img_path.exists():
            print(f"✗ 图像文件缺失: {img_path}")
            missing_count += 1
        
        # 检查mask文件
        seg_path = target_path / 'segmentation' / row['study'] / row['view'] / f"{row['dicom_uuid']}.nii.gz"
        if not seg_path.exists():
            print(f"✗ Mask文件缺失: {seg_path}")
            missing_count += 1
    
    if missing_count == 0:
        print(f"✓ 所有文件完整")
        return True
    else:
        print(f"✗ 缺失 {missing_count} 个文件")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="转换Segment_DATA数据集")
    parser.add_argument('--source', default='/home/ubuntu/Segment_DATA',
                       help='源数据目录')
    parser.add_argument('--target', default='/home/ubuntu/my_organized_dataset',
                       help='目标数据目录')
    parser.add_argument('--verify', action='store_true',
                       help='仅验证已转换的数据')
    args = parser.parse_args()
    
    if args.verify:
        # 仅验证
        verify_conversion(args.target)
    else:
        # 转换
        try:
            df = convert_dataset(args.source, args.target)
            
            # 自动验证
            verify_conversion(args.target)
            
            print(f"\n数据集统计:")
            print(df.groupby('my_split').size())
            print(f"\n可以开始训练了：")
            print(f"  cd /home/ubuntu/RL4Seg3D")
            print(f"  python3 rl4seg3d/auto_iteration.py")
            
        except Exception as e:
            print(f"\n✗ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

