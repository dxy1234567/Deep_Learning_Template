"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader.Dataset import KittiDepthDataset
num_worker = 8

def KittiDataLoader(params):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    num_worker = 8

    image_datasets = {}
    dataloaders = {}

    norm_size = params['data_normalize_factor']
    data_path = params['dir_ds']
    # crop = transforms.CenterCrop((1056, 1920))
    crop = transforms.CenterCrop((480, 832))



    ###### Training Set ######
    image_datasets['train'] = KittiDepthDataset(data_path, setname='train', norm_size=norm_size, transform=crop, flip=False)

    # Select the desired number of images from the training set
    if params['train_on'] != 'full':
        image_datasets['train'].data = image_datasets['train'].data[0:params['train_on']]  # file directions
        image_datasets['train'].gt = image_datasets['train'].gt[0:params['train_on']]

    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=num_worker)
    ###### Test Set ######
    image_datasets['test'] = KittiDepthDataset(data_path, setname='test', norm_size=norm_size, transform=crop, flip=False)

    dataloaders['test'] = DataLoader(image_datasets['test'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=num_worker)

    ###### Validation Set ######
    image_datasets['val'] = KittiDepthDataset(data_path, setname='val', norm_size=norm_size, transform=crop, flip=False)
    
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=True, batch_size=params['val_batch_sz'],
                                    num_workers=num_worker)

    return dataloaders







