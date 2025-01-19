import os
import cv2
import numpy as np
from tqdm import tqdm

def convert_65535_to_0_inplace(directory):
    """
    原地将目录中的图像中值为65535的像素转换为0。
    """
    # 获取目录下所有图像文件
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.tif'))]

    print(f"Processing directory: {directory}")
    for filename in tqdm(files, desc="Processing Images"):
        file_path = os.path.join(directory, filename)

        # 读取图像
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is not None:
            # 将值为65535的像素转换为0
            image[image == 65535] = 0

            # 原地保存图像
            cv2.imwrite(file_path, image)
        else:
            print(f"Failed to load image: {filename}")

# 设置需要处理的目录
train_depth_dir = "/data/dataset_blender_1/train/depth"
train_gt_dir = "/data/dataset_blender_1/train/gt"
test_depth_dir = "/data/dataset_blender_1/test/depth"
test_gt_dir = "/data/dataset_blender_1/test/gt"

# 原地转换训练集 depth 和 gt
convert_65535_to_0_inplace(train_depth_dir)
convert_65535_to_0_inplace(train_gt_dir)

# 原地转换测试集 depth 和 gt
convert_65535_to_0_inplace(test_depth_dir)
convert_65535_to_0_inplace(test_gt_dir)
