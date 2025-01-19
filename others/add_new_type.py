"""
    在数据集中添加新的数据类型，如depth_gt_21。
"""

import os
import shutil

def add_new_data_type(dataset_dir, raw_data_dir, new_data_type):
    """
    从原始数据中提取新数据类型文件到数据集目录中。

    :param dataset_dir: 数据集文件夹路径
    :param raw_data_dir: 原始数据文件夹路径
    :param new_data_type: 新的数据类型（如 depth_gt_21）
    """
    subsets = ['train', 'val', 'test']

    for subset in subsets:
        # 数据集中的图片路径
        subset_dir = os.path.join(dataset_dir, subset, "depth")
        target_dir = os.path.join(dataset_dir, subset, new_data_type)
        os.makedirs(target_dir, exist_ok=True)

        for file_name in os.listdir(subset_dir):
            if not file_name.endswith(".png"):
                continue

            # 解析文件名 xx_yyyyyy.png
            seq, frame = file_name.split('_')
            frame = frame.split('.')[0]  # 去掉 .png

            # 构造原始数据文件路径
            raw_file_path = os.path.join(raw_data_dir, seq, new_data_type, f"{frame}.png")

            # 检查文件是否存在
            if os.path.exists(raw_file_path):
                # 将文件复制到目标目录，并保持命名一致
                target_file_path = os.path.join(target_dir, file_name)
                shutil.copy(raw_file_path, target_file_path)
                print(f"复制 {raw_file_path} -> {target_file_path}")
            else:
                print(f"原始文件不存在: {raw_file_path}")


# 参数配置
dataset_directory = "/data/KITTI_to_DC/dataset_1_cropped/"  # 数据集目录
raw_data_directory = "/data/KITTI_to_DC/"  # 原始数据目录
new_data_type_name1 = "depth_gt_21"  # 新的数据类型
new_data_type_name2 = "depth_gt_41"  # 新的数据类型

# 执行函数
add_new_data_type(dataset_directory, raw_data_directory, new_data_type_name1)
