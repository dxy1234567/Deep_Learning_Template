import os

def delete_files_with_conditions(dataset_dir, raw_data_dir):
    """
    根据以下条件删除数据集中的文件：
    1. 删除属于 "00" 序列的文件。
    2. 删除序号在 0-99 范围内的文件。
    3. 删除属于原始序列的最后 100 个序号的文件。

    :param dataset_dir: 数据集根目录
    :param raw_data_dir: 原始数据根目录，用于确定每个序列的最大帧编号
    """
    subsets = ['train', 'val', 'test']
    categories = ['depth', 'depth_gt', 'gray']

    # 记录每个序列的最大帧编号
    sequence_max_frames = {}

    # 遍历原始数据文件，确定每个序列的最大帧编号
    for seq in os.listdir(raw_data_dir):
        if not seq.startswith("0"):  # 跳过非以 0 开头的序列
            continue

        seq_path = os.path.join(raw_data_dir, seq)
        if not os.path.isdir(seq_path):
            continue

        # 获取序列内所有帧编号
        frame_numbers = []
        for file_name in os.listdir(os.path.join(seq_path, 'depth')):
            if file_name.endswith(".png"):
                frame = int(file_name.split('.')[0])  # 提取帧编号
                frame_numbers.append(frame)

        if frame_numbers:
            sequence_max_frames[seq] = max(frame_numbers)

    # 遍历数据集目录，删除符合条件的文件
    for subset in subsets:
        subset_dir = os.path.join(dataset_dir, subset)
        for category in categories:
            category_dir = os.path.join(subset_dir, category)
            if not os.path.exists(category_dir):
                continue

            for file_name in os.listdir(category_dir):
                if not file_name.endswith(".png"):
                    continue

                # 解析文件名 xx_yyyyyy.png
                seq, frame = file_name.split('_')
                frame = int(frame.split('.')[0])  # 提取帧编号

                # 条件 1：删除 "00" 序列的文件
                if seq == "00":
                    file_path = os.path.join(category_dir, file_name)
                    os.remove(file_path)
                    print(f"删除文件（条件 1 - 00 序列）：{file_path}")
                    continue

                # 条件 2：删除序号在 0-99 范围内的文件
                if 0 <= frame <= 99:
                    file_path = os.path.join(category_dir, file_name)
                    os.remove(file_path)
                    print(f"删除文件（条件 2 - 序号 0-99）：{file_path}")
                    continue

                # 条件 3：删除原始序列的最后 100 个序号的文件
                max_frame = sequence_max_frames.get(seq)
                if max_frame and frame >= max_frame - 99:
                    file_path = os.path.join(category_dir, file_name)
                    os.remove(file_path)
                    print(f"删除文件（条件 3 - 最后 100 个序号）：{file_path}")

# 参数配置
dataset_directory = "/data/KITTI_to_DC/dataset_1_cropped/"  # 数据集根目录
raw_data_directory = "/data/KITTI_to_DC/"  # 原始数据根目录

# 执行函数
delete_files_with_conditions(dataset_directory, raw_data_directory)
