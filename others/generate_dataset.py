import os
import random
import shutil

def extract_random_frames_with_types(sequences, sequences_dir, output_dir, test_ratio=0.1, val_ratio=0.05):
    """
        从每个序列中随机抽取num_frames个帧，并确保depth, depth_gt, gray三种数据对应。

        Params: 
            base_dir: 存放序列的根目录，每个序列有depth, depth_gt, gray三个子文件夹
            output_dir: 测试集输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    types_data = ['depth', 'depth_gt', 'gray', 'depth_gt_21', 'depth_gt_41', 'depth_gt_61']

    for sequence in sequences:
        # 获取三个类型的文件夹路径，在每个序列中
        type_dirs = {t: os.path.join(sequences_dir, sequence, t) for t in types_data}
        
        # 确保三个子目录都存在
        for t, t_dir in type_dirs.items():
            if not os.path.isdir(t_dir):
                print(f"目录 {t_dir} 不存在！")
                return
        
        # 获取depth文件夹中的所有帧（以gt为基准【gt少】）
        frame_files = sorted([f for f in os.listdir(type_dirs['depth_gt_21']) if f.endswith(('.jpg', '.png'))])  ### 
        frames_size = len(frame_files)

        test_size = int(frames_size * test_ratio)
        val_size = int(frames_size * val_ratio)

        # 随机选择测试集
        test_frames = random.sample(frame_files, test_size)
        remaining_frames = list(set(frame_files) - set(test_frames))
        
        # 在剩余数据中随机选择验证集
        val_frames = random.sample(remaining_frames, val_size)
        train_frames = list(set(remaining_frames) - set(val_frames))
        
        # 定义数据集的输出目录
        subsets = {'train': train_frames, 'val': val_frames, 'test': test_frames}
        
        # 复制每个子集的数据到对应目录
        for subset_name, frames in subsets.items():
            for t in types_data:
                subset_output_dir = os.path.join(output_dir, subset_name, t)  # dataset/train/depth/.
                os.makedirs(subset_output_dir, exist_ok=True)

                # 复制选中的帧到输出目录
                for frame in frames:
                    src = os.path.join(type_dirs[t], frame)
                    dst = os.path.join(subset_output_dir, sequence + "_" + frame)
                    shutil.copy(src, dst)

        ###############
                
        print(f"从序列{sequence}中随机抽取了数据保存到 {output_dir}")

# 使用示例
sequence_dir = '/data/KITTI_to_DC/'  # 原始序列的根目录
output_dir = '/data/KITTI_to_DC/dataset_1_frames/'      # 测试集输出目录


# sequences = {"01"}
sequences = {"04", "05", "06", "07", "08"}

extract_random_frames_with_types(sequences, sequence_dir, output_dir)