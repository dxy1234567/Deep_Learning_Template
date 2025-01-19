import os
import random
import shutil

def add_dataset(sequences, sequences_dir, output_dir, set='train'):
    """
        把数据读读入到某一类型的数据集当中。

        Params: 
            base_dir: 存放序列的根目录，每个序列有depth, depth_gt, gray三个子文件夹
            output_dir: 测试集输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    types_data = ['depth', 'depth_gt', 'gray']

    for sequence in sequences:
        # 获取三个类型的文件夹路径，在每个序列中
        type_dirs = {t: os.path.join(sequences_dir, sequence, t) for t in types_data}
            
        # 确保三个子目录都存在  
        for t, t_dir in type_dirs.items():
            if not os.path.isdir(t_dir):
                print(f"目录 {t_dir} 不存在！")
                return
        
        frame_file = sorted(os.listdir(type_dirs['depth_gt']))
        
        for t in types_data:
            subset_output_dir = os.path.join(output_dir, set, t)  # dataset/train/depth/.
            os.makedirs(subset_output_dir, exist_ok=True)

            # 复制选中的帧到输出目录
            for frame in frame_file:
                src = os.path.join(type_dirs[t], frame)
                dst = os.path.join(subset_output_dir, sequence + "_" + frame)
                shutil.copy(src, dst)

        ###############
                
        print(f"从序列中随机抽取了数据保存到 {output_dir}")

# 使用示例
sequence_dir = '/data/gml_to_DC/'  # 原始序列的根目录
output_dir = '/data/gml_to_DC/dataset_04/'      # 测试集输出目录


# sequences = {"01"}
sequences = {"04"}

add_dataset(sequences, sequence_dir, output_dir)