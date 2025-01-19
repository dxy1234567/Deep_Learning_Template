import os
import cv2
import numpy as np
import sys
sys.path.append('.')
from utils.util import print_progress


def stack_images_vertically(folder1, folder2, folder3, folder4, output_folder):
    """
        将四个文件夹中对应的图片上下拼接，并保存拼接后的新图片。

        :param folder1: 第一个文件夹路径
        :param folder2: 第二个文件夹路径
        :param folder3: 第三个文件夹路径
        :param folder4: 第四个文件夹路径
        :param output_folder: 拼接后的图像保存路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取四个文件夹中的所有图片文件，按名称排序
    images1 = sorted([os.path.join(folder1, img) for img in os.listdir(folder1) if img.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    images2 = sorted([os.path.join(folder2, img) for img in os.listdir(folder2) if img.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    images3 = sorted([os.path.join(folder3, img) for img in os.listdir(folder3) if img.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    images4 = sorted([os.path.join(folder4, img) for img in os.listdir(folder4) if img.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])

    # 确保四个文件夹的图像数量一致
    assert len(images1) == len(images2) == len(images3) == len(images4), "图片数量不一致！"

    N = len(images1)

    image = cv2.imread(images4[0])

    shape = (image.shape[1], image.shape[0])


    # 遍历所有图片并进行拼接
    for idx, (img1, img2, img3, img4) in enumerate(zip(images1, images2, images3, images4)):
        # 读取图片
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2)
        image3 = cv2.imread(img3)
        image4 = cv2.imread(img4)

        image1 = cv2.resize(image1, shape)
        image2 = cv2.resize(image2, shape)
        image3 = cv2.resize(image3, shape)

        # 确保四张图片的宽度一致（如果不一致，可以进行调整）
        if image1.shape[1] != image2.shape[1] or image2.shape[1] != image3.shape[1] or image3.shape[1] != image4.shape[1]:
            raise ValueError("四张图片的宽度不一致，请调整图片大小！")

        max_image1 = np.max(image1)
        max_image2 = np.max(image2)

        image1 = image1 / max_image1 * 255
        image2 = image2 / max_image2 * 255

        # 将四张图片上下拼接
        stacked_image = np.vstack((image1, image2, image3, image4))

        # 保存拼接后的图片
        output_img_path = os.path.join(output_folder, os.path.basename(img1))
        cv2.imwrite(output_img_path, stacked_image)
        print(f"已保存拼接图像：{output_img_path}")
        print_progress(idx, N)

# 使用示例
folder1 = '/data/gml_to_DC/dataset_02/test/depth/'  # 第一个文件夹路径
folder2 = '/data/gml_to_DC/dataset_02/test/depth_gt/'  # 第二个文件夹路径
folder3 = '/data/gml_to_DC/dataset_02/test/gray/'  # 第三个文件夹路径
folder4 = 'output/gml_udgd_prdct'  # 第四个文件夹路径
output_folder = 'output/gml_udgd'  # 拼接后的图像保存路径


# 调用函数
stack_images_vertically(folder1, folder2, folder3, folder4, output_folder)
