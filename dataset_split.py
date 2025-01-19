import os
import random
import shutil
from tqdm import tqdm  # 引入 tqdm

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    将数据集分为 train、val 和 test 子集，并使用 tqdm 显示处理进度。
  
    :param dataset_dir: 数据集路径，包含 'gt', 'gray', 'depth' 三个子文件夹
    :param output_dir: 输出路径，将生成 'train', 'val', 'test' 三个子文件夹
    :param train_ratio: 训练集比例，默认0.8
    :param val_ratio: 验证集比例，默认0.1（测试集比例为 1 - train_ratio - val_ratio）
    """
    # 确保输出目录存在
    subsets = ["train", "val", "test"]
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset), exist_ok=True)
        for folder in ["gt", "gray", "depth_2_0105", "depth_3_line32"]:
            os.makedirs(os.path.join(output_dir, subset, folder), exist_ok=True)

    # 遍历 'gt', 'gray', 'depth' 文件夹，确保文件名一一对应
    gt_files = sorted(os.listdir(os.path.join(dataset_dir, "gt")))
    gray_files = sorted(os.listdir(os.path.join(dataset_dir, "gray")))
    depth_1_files = sorted(os.listdir(os.path.join(dataset_dir, "depth_2_0105")))
    depth_2_files = sorted(os.listdir(os.path.join(dataset_dir, "depth_3_line32")))

    # 确保三者文件数量一致
    assert len(gt_files) == len(gray_files) == len(depth_1_files) == len(depth_2_files), "GT, Gray 和 Depth 文件数量不一致，请检查数据！"

    # 随机打乱文件列表
    file_count = len(gt_files)
    indices = list(range(file_count))
    random.shuffle(indices)

    # 计算划分点
    train_count = int(file_count * train_ratio)
    val_count = int(file_count * val_ratio)
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    # 定义数据复制函数
    def copy_files(file_list, src_folder, dest_folder, indices, subset_name):
        with tqdm(total=len(indices), desc=f"Processing {subset_name}/{src_folder}", unit="file") as pbar:
            for i, idx in enumerate(indices):
                src_file = os.path.join(dataset_dir, src_folder, file_list[idx])
                dest_file = os.path.join(dest_folder, f"{i + 1:08d}.png")  # 重命名为 00000001.png
                shutil.copy2(src_file, dest_file)
                pbar.update(1)  # 更新进度条

    # 分别处理 train, val, test 子集
    for subset, subset_indices in zip(subsets, [train_indices, val_indices, test_indices]):
        for folder in ["gt", "gray", "depth_2_0105", "depth_3_line32"]:
            copy_files(gt_files if folder == "gt" else gray_files if folder == "gray" else depth_1_files if folder == "depth_2_0105" else depth_2_files,
                       folder, os.path.join(output_dir, subset, folder), subset_indices, subset)

    # 输出结果
    print(f"数据集划分完成！训练集: {len(train_indices)} 张，验证集: {len(val_indices)} 张，测试集: {len(test_indices)} 张")

# 默认参数
dataset_dir = "/data/dataset_total_2"  # 原始数据集路径
output_dir = "/data/dataset_blender_3"  # 输出数据集路径
train_ratio = 0.9  # 默认训练集比例
val_ratio = 0.05  # 默认验证集比例

# 运行函数
split_dataset(dataset_dir, output_dir, train_ratio, val_ratio)
