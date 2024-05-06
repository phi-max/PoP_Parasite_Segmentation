# %% Imports
import os
import unet_framework
import torch
import numpy as np
import albumentations

from unet_framework.utils.general_utils import get_IDs_of_dir
from unet_framework.utils.augmentations import CenterCrop, Dense_Target, AlbuSeg3d, ExpandDim, RandomFlip, Normalize01, One_hot_encoding
from unet_framework.unet.unet import UNet
from unet_framework.eval import metrics

# %% Save directory

save_dir = '/data/s339697/Framework_U-Net/master_thesis/Results/'  # location where the training files will saved to
save_folder_name = 'UNET_MSE'  # folder name
save_root = os.path.join(save_dir, save_folder_name)

# Save config
save_dir_config = '/data/s339697/Training/MT/config_files'  # location where the config file will be saved to
name_config = 't_config.py'  # config name
save_name_config = os.path.join(save_dir_config, name_config)

# %% Config file
import unet_framework.data.Microtubules.BG_CM_MT.Training_data.Size_24x200x200.Input as input_dir_training
import unet_framework.data.Microtubules.BG_CM_MT.Training_data.Size_24x200x200.Target as target_dir_training

import unet_framework.data.Microtubules.BG_CM_MT.Test_Validation_data.Input as input_dir_validation
import unet_framework.data.Microtubules.BG_CM_MT.Test_Validation_data.Target as target_dir_validation

# Input and target directories
input_dir_training = input_dir_training.__path__._path[0]  # location of the input files (Training)
target_dir_training = target_dir_training.__path__._path[0]  # location of the target files (Training)

input_dir_validation = input_dir_validation.__path__._path[0]  # location of the input files (Validation)
target_dir_validation = target_dir_validation.__path__._path[0]  # location of the target files (Validation)

# File extension
input_ext = '.tif'  # input extension
target_ext = '.tif'  # target extension

# Input and target IDs
train_input_IDs = get_IDs_of_dir(input_dir_training, ext=input_ext)  # list of input files (Training)
train_target_IDs = get_IDs_of_dir(target_dir_training, ext=target_ext)  # list of target files (Training)

valid_input_IDs = get_IDs_of_dir(input_dir_validation, ext=input_ext)  # list of input files (Validation)
valid_target_IDs = get_IDs_of_dir(target_dir_validation, ext=target_ext)  # list of target files (Validation)

# Cropping sizes
input_size = (24, 200, 200)
target_size = (24, 200, 200)

# Training transformations
transform_train_args = [
    CenterCrop(input_size=input_size, target_size=target_size),  # Center-crop to input and target size
    Dense_Target(),  # Ensure a dense integer representation
    AlbuSeg3d(albu=albumentations.ShiftScaleRotate(p=0.75,  # Wrapper for albumentations
                                                   rotate_limit=180,
                                                   scale_limit=0,
                                                   shift_limit=0.0625,
                                                   interpolation=3)),
    ExpandDim(transform_input=True, transform_target=False),
    # Make it suitable for transformations [C, D, H, W] or [C, H, W]
    RandomFlip(ndim_spatial=3),  # Random 90° Flip along all spatial axes
    One_hot_encoding(),
    Normalize01()  # Linear Scaling [0-1]
]

transform_train = unet_framework.utils.augmentations.Compose(transform_train_args)

# Validation transformations
transform_valid_args = [
    Dense_Target(),  # Ensure a dense integer representation
    ExpandDim(),  # Make it suitable for transformations [C, D, H, W] or [C, H, W]
    One_hot_encoding(),
    Normalize01()  # Linear Scaling [0-1]
]

transform_valid = unet_framework.utils.augmentations.Compose(transform_valid_args)

# Seed
random_seed = 0

# Model
out_channels = 3
model = UNet(in_channels=1,
             out_channels=out_channels,
             n_blocks=4,
             start_filts=32,
             normalization='instance',
             dim=3,
             conv_mode='same',
             activation='elu',
             weight_ini='uniform'
             )

# Loss
criterion1 = torch.nn.MSELoss()  # MSE loss

# Default settings
epochs = 1
update = 1
epoch = 0

train_batch_size = 1
valid_batch_size = 1

train_shuffle = True
valid_shuffle = True

input_dtype = torch.float32
target_dtype = torch.float32

save_iteration_plot = True

# Optimizer
optim = 'Adam'
optimizer_params = {'lr': 1e-4,
                    'weight_decay': 5e-05
                    }
# Learning rate scheduler
lr_sched = 'ReduceLROnPlateau'
lr_sched_parameters = {'factor': 0.75,
                       'patience': 75,
                       'min_lr': 1e-6
                       }

# Evaluation metrics
metrics_train_valid = {}
metrics_train_valid.update({f'Accuracy': [metrics.Accuracy(index=i) for i in range(out_channels)]})
metrics_train_valid.update({f'Accuracy_mean': metrics.Accuracy()})
metrics_train_valid.update({f'IoU': [metrics.IoU(index=i) for i in range(out_channels)]})
metrics_train_valid.update({f'IoU_mean': metrics.IoU()})
metrics_train_valid.update({f'DSC': [metrics.DSC(index=i) for i in range(out_channels)]})
metrics_train_valid.update({f'DSC_mean': metrics.DSC()})

# Early stopping
early_stopping_params = {'mode': 'min', 'patience': 150, 'crit': 'valid_loss_mean'}

# Save model criteria
save_best_model_params = {'save_mode': 'max', 'save_crit': 'valid_DSC_mean', 'save_epochs': 10}

# Config file building
config_file = {
    'save_root': save_root,
    'random_seed': random_seed,
    'train_IDs': {
        'input': train_input_IDs,
        'target': train_target_IDs
    },
    'valid_IDs': {
        'input': valid_input_IDs,
        'target': valid_target_IDs
    },
    'transforms_train': transform_train,
    'transforms_valid': transform_valid,
    'train_batch_size': train_batch_size,
    'valid_batch_size': valid_batch_size,
    'train_shuffle': train_shuffle,
    'valid_shuffle': valid_shuffle,
    'input_dtype': input_dtype,
    'target_dtype': target_dtype,
    'model': model,
    'criterion': criterion1,
    'epochs': epochs,
    'update': update,
    'epoch': epoch,
    'optimizer': {optim: optimizer_params},
    'lr_scheduler': {lr_sched: lr_sched_parameters},
    'metrics': metrics_train_valid,
    'save_iteration_plot': save_iteration_plot,
    'early_stopping_params': early_stopping_params,
    'save_best_model_params': save_best_model_params,
}

# Save config file
torch.save(config_file, save_name_config)
print(f'config_file saved at {save_name_config}')

# %% Start training

vglrun = False  # Using vglrun
python_loc = '/data/s339697/miniconda3_2/envs/Frame/bin/python'  # python location
from unet_framework import train

train_file = train.__file__

epochs = 1000
update = 25

if vglrun:
    os.system(
        f'vglrun {python_loc} {train_file} {save_name_config} --epochs {epochs} --update {update} --use-cache --no-question')
else:
    os.system(
        f'{python_loc} {train_file} {save_name_config} --epochs {epochs} --update {update} --use-cache --no-question')
