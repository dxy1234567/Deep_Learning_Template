import os
import sys
import importlib
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.saveTensorToImage import *
from torch.optim import lr_scheduler

from dataloader.DataLoaders import *
from modules.losses import *


# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn

if __name__ == '__main__':
#     print(torch.cuda.is_available())
    cudnn.enabled = True
    cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.cuda.manual_seed(1)
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', dest='mode', default='eval', help='"eval" or "train" mode')
    parser.add_argument('-exp', action='store', dest='exp', default='blender_1',
                        help='Experiment name as in workspace directory')
    #parser.add_argument('-chkpt', action='store', dest='chkpt', default=50,  nargs='?',   # None or number
    parser.add_argument('-chkpt', action='store', dest='chkpt', default="workspace/blender_1/depth2_epoch27.pth.tar",
                        help='Checkpoint number to load')

    parser.add_argument('-set', action='store', dest='set', default='test', type=str, nargs='?',
                        help='Which set to evaluate on "val", "selval" or "test"')
    args = parser.parse_args()

    # Path to the workspace directory
    training_ws_path = 'workspace'
    exp = args.exp
    exp_dir = os.path.join(training_ws_path, exp)

    # Add the experiment's folder to python path
    sys.path.append(exp_dir)

    # Read parameters file
    with open(os.path.join(exp_dir, 'params.json'), 'r') as fp:
        params = json.load(fp)
    params['gpu_id'] = "0"

    # Use GPU or not
    #device = torch.device("cuda:" + str(params['gpu_id']) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + params['gpu_id'] if torch.cuda.is_available() else "cpu")

    # Dataloader
    data_loader = params['data_loader'] if 'data_loader' in params else 'KittiDataLoader'
    dataloaders = eval(data_loader)(params)

    # Import the network file
    f = importlib.import_module('network_' + exp)
    model = f.network().to(device)#pos_fn=params['enforce_pos_weights']
    model = nn.DataParallel(model)

    # Import the trainer
    t = importlib.import_module('trainers.' + params['trainer'])

    if args.mode == 'train':
        mode = 'train'  # train    eval
        sets = ['train']  # train  selval
    elif args.mode == 'eval':
        mode = 'eval'  # train    eval
        sets = [args.set]  # train  selval

    # Objective function
    objective = locals()[params['loss']]()

    # Optimize only parameters that requires_grad
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # The optimizer
    optimizer = getattr(optim, params['optimizer'])(parameters, lr=params['lr'],
                                                        weight_decay=params['weight_decay'])

    # Decay LR by a factor of 0.1 every exp_dir7 epochs
    # lr_decay = lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_decay_step'], gamma=params['lr_decay']) #
    lr_decay = lr_scheduler.StepLR(optimizer, step_size=params['lr_decay_step'], gamma=params['lr_decay'])

    mytrainer = t.KittiDepthTrainer(model, params, optimizer, objective, lr_decay, dataloaders,
                                        workspace_dir=exp_dir, sets=sets, use_load_checkpoint=args.chkpt)
    
    # --- Configure Done ---

    # --- Data Loading ---

    depth_path = '/data/dataset_blender_3/test/depth_2_0105/00000023.png'
    gray_path = '/data/dataset_blender_3/test/gray/00000023.png'
    gt_path = '/data/dataset_blender_3/test/gt/00000023.png'

    input_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    input_gray = cv2.imread(gray_path, cv2.IMREAD_UNCHANGED)

    input_depth, input_gray = mytrainer.preprocess(input_depth, input_gray, params['data_normalize_factor'])

    output = mytrainer.predict(input_depth, input_gray)

    save_one_tensor_to_image(output, 'output/2.png', params['data_normalize_factor'])
