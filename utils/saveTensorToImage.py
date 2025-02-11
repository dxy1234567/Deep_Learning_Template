"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""


import torch
import os
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np


# This function takes a 4D tensor in the form NxCxWxH and save it to images according to idxs
def saveTensorToImage(t, idxs, save_to_path):
    if os.path.exists(save_to_path) == False:
        os.mkdir(save_to_path)

    for i in range(t.size(0)):
        im = t[i, :, :, :].detach().data.cpu().numpy()
        im = np.transpose(im, (1, 2, 0)).astype(np.uint16)
        '''
        imout = np.zeros((374,1238,1))
        imout[374-352:,:,:] = im[:,5:-5,:]
        '''
        imout = im

        cv2.imwrite(os.path.join(save_to_path, str(idxs[i].data.cpu().numpy()).zfill(10) + '.png'), imout,
                    [cv2.IMWRITE_PNG_COMPRESSION, 4])

def save_one_tensor_to_image(image_tensor, save_to_path, norm_size):
    """
        带后处理
    """
    image_tensor = image_tensor * norm_size

    img = image_tensor[0, :, :, :].detach().data.cpu().numpy()
    img = np.transpose(img, (1, 2, 0)).astype(np.uint16)
    cv2.imwrite(save_to_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
