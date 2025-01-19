import os

def sync_folder_files(folder1, folder2):
    """
    检查两个文件夹中的文件数，如果文件数不同，
    以文件数较少的文件夹为基准，删除文件数较多文件夹中多余的文件。

    :param folder1: 文件夹路径1
    :param folder2: 文件夹路径2
    """
    # 获取两个文件夹中的文件列表（仅限文件）
    files1 = set(f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f)))
    files2 = set(f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f)))

    # 比较文件数量
    if len(files1) == len(files2):
        print("两个文件夹的文件数相同，无需操作。")
        return

    # 确定基准文件夹和多余文件夹
    if len(files1) < len(files2):
        base_files, extra_files = files1, files2
        extra_folder = folder2
    else:
        base_files, extra_files = files2, files1
        extra_folder = folder1

    # 找到需要删除的文件（多余文件）
    files_to_delete = extra_files - base_files

    # 删除多余文件
    for file_name in files_to_delete:
        file_path = os.path.join(extra_folder, file_name)
        os.remove(file_path)
        print(f"已删除文件：{file_path}")

    print(f"已完成同步操作，以文件数较少的文件夹为基准。")


folder1_path = "/data/KITTI_to_DC/dataset_1_frames_cropped/train/depth/"  # 文件夹1路径
folder2_path = "/data/KITTI_to_DC/dataset_1_frames_cropped/train/depth_gt_21/"  # 文件夹2路径

sync_folder_files(folder1_path, folder2_path)
