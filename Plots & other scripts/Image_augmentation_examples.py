# %% Imports

import albumentations
import torch
from torch.utils import data

from unet_framework.utils.augmentations import CenterCrop, ExpandDim, AlbuSeg3d, Dense_Target, Normalize01, Compose, RandomFlip
from unet_framework.utils.customDataSets import CustomDataSet
from unet_framework.utils.general_utils import get_IDs_of_dir

import numpy as np
from skimage.io import imsave
import os

import unet_framework.data.Microtubules.BG_CM_MT.Training_data.Size_24x200x200.Input as input_dir_training
import unet_framework.data.Microtubules.BG_CM_MT.Training_data.Size_24x200x200.Target as target_dir_training

# Input and target directories
input_dir_training = input_dir_training.__path__._path[0]  # location of the input files (Training)
target_dir_training = target_dir_training.__path__._path[0]  # location of the target files (Training)

# File extension
input_ext = '.tif'  # input extension
target_ext = '.tif'  # target extension

# Input and target IDs
train_input_IDs = get_IDs_of_dir(input_dir_training, ext=input_ext)  # list of input files (Training)
train_target_IDs = get_IDs_of_dir(target_dir_training, ext=target_ext)  # list of target files (Training)

train_input_IDs = [train_input_IDs[0]]
train_target_IDs = [train_target_IDs[0]]


def postprocess(inp):
    from unet_framework.utils.general_utils import re_normalize
    inp = inp.data.numpy()
    inp = inp[0]  # remove batch dim
    inp = inp[0]  # remove channel dim
    inp = inp[11]  # select slice
    inp = re_normalize(inp)  # re_normalize to range [0, 255]
    inp = inp.astype(np.uint8)

    return inp


# Cropping sizes
input_size = (24, 200, 200)
target_size = (24, 200, 200)

# Cropping sizes
input_size = (24, 200, 200)
target_size = (24, 200, 200)



transforms = []

# Random flip
transform_train_args = [
    CenterCrop(input_size=input_size, target_size=target_size),  # Center-crop to input and target size
    Dense_Target(),
    ExpandDim(transform_input=True, transform_target=False),
    RandomFlip(ndim_spatial=3),  # Random 90° Flip along all spatial axes
    Normalize01()
]

transforms.append(transform_train_args)

# Random shift
transform_train_args = [
    CenterCrop(input_size=input_size, target_size=target_size),  # Center-crop to input and target size
    Dense_Target(),
    AlbuSeg3d(albu=albumentations.ShiftScaleRotate(p=1.0,
                                                   rotate_limit=0,
                                                   scale_limit=0,
                                                   shift_limit=0.0625,
                                                   interpolation=3)),
    ExpandDim(transform_input=True, transform_target=False),
    Normalize01()
]

transforms.append(transform_train_args)

# Random rotate
transform_train_args = [
    CenterCrop(input_size=input_size, target_size=target_size),  # Center-crop to input and target size
    Dense_Target(),
    AlbuSeg3d(albu=albumentations.ShiftScaleRotate(p=1.0,
                                                   rotate_limit=180,
                                                   scale_limit=0,
                                                   shift_limit=0,
                                                   interpolation=3)),
    ExpandDim(transform_input=True, transform_target=False),
    # Make it suitable for transformations [C, D, H, W] or [C, H, W]
    # RandomFlip(ndim_spatial=3),  # Random 90° Flip along all spatial axes
    Normalize01()
]

transforms.append(transform_train_args)


from sklearn.model_selection import ParameterGrid

parameters = {
    'transform': transforms
}

parameter_grid = ParameterGrid(parameters)
# %% Config file


for i, param_set in enumerate(parameter_grid):
    transform_train = Compose(param_set['transform'])

    DataSet_training = CustomDataSet(input_list_IDs=train_input_IDs,
                                     target_list_IDs=train_target_IDs,
                                     transform=transform_train,
                                     input_dtype=torch.float32,
                                     target_dtype=torch.long,
                                     )

    DataLoader_training = data.DataLoader(DataSet_training,
                                          shuffle=True,
                                          batch_size=1,
                                          )

    x, y = next(iter(DataLoader_training))
    x = postprocess(x)
    y = y[0]
    y = y[11]
    y = y.data.numpy()

    from unet_framework.utils.visual_utils import show_img

    show_img(x)

    # imsave(os.path.join('/data/s339697/master_thesis/Images', f'augment_{i+1}.png'), x)

