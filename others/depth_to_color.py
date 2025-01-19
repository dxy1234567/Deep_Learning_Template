import numpy as np
import cv2
import os
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

def depth_to_colormap_dir(dir_imgs, dir_colormap):
    # 创建保存伪彩色图的目录
    os.makedirs(dir_colormap, exist_ok=True)

    # 获取所有图像文件
    imgs_list = os.listdir(dir_imgs)
    imgs_list = [img for img in imgs_list if img.endswith('png')]  # 只处理 PNG 文件

    # 使用 tqdm 显示进度
    for img in tqdm(imgs_list, desc="Processing images"):
        path_img = os.path.join(dir_imgs, img)

        # 读取深度图
        image = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
        
        # 获取图像的最大值并避免除以零
        max_image = np.max(image)
        if max_image == 0:
            print(f"Warning: {img} has a maximum value of 0, skipping...")
            continue

        # 转换为 0-255 范围的灰度图
        image_255 = image / max_image * 255.0
        image_255 = np.uint8(image_255)

        # 生成伪彩色图
        image_colormap = cv2.applyColorMap(image_255, cv2.COLORMAP_JET)

        # 保存伪彩色图
        path_colormap = os.path.join(dir_colormap, img)
        cv2.imwrite(path_colormap, image_colormap)

def depth_to_colormap(image_path, output_path):
    # 获取图像的最大值并避免除以零
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    mask_invalid = (image == 0)

    max_image = np.max(image)
    if max_image == 0:
        print("Warning: image has a maximum value of 0, skipping...")
        return None
    # 转换为 0-255 范围的灰度图
    image_255 = image / max_image * 255.0
    image_255 = np.uint8(image_255)
    # 生成伪彩色图
    image_colormap = cv2.applyColorMap(image_255, cv2.COLORMAP_JET)

    image_colormap[mask_invalid] = [0, 0, 0]
    

    cv2.imwrite(output_path, image_colormap)



# # # 输入目录和输出目录(处理文件夹)
# dir_imgs = 'workspace/blender_1/test_output_epoch_1'
# dir_colormap_output = 'output/blender_1_colormap3_depth_3'

# depth_to_colormap_dir(dir_imgs, dir_colormap_output)

# # 处理单张图片
image_path = 'output/2.png'

# output_path = os.path.join('output', os.path.basename(image_path))
output_path = os.path.join('output', '2_colormap.png')
depth_to_colormap(image_path, output_path)