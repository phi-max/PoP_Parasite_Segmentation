# %% Imports
import os

import numpy as np
import torch
from unet_framework.eval import metrics
from unet_framework.utils.general_utils import get_IDs_of_dir

# %% Directory
dir = """/master_thesis/Results/UNET_Levels/"""  # Specifiy the training folder here
sub_dirs = os.listdir(dir)
sub_dirs = sorted(sub_dirs)

# %% Predict (validation data) every experiment
from unet_framework.predict.predict import simple_predict
from skimage.io import imsave, imread

# Device
device_name = 'cuda'
device = torch.device(device_name)

import unet_framework.data.Microtubules.BG_CM_MT.Test_Validation_data.Input as input_dir_validation
import unet_framework.data.Microtubules.BG_CM_MT.Test_Validation_data.Target as target_dir_validation

# Load input & target
input_dir_validation = input_dir_validation.__path__._path[0]  # location of the input files (Validation)
target_dir_validation = target_dir_validation.__path__._path[0]  # location of the target files (Validation)

# File extension
input_ext = '.tif'  # input extension
target_ext = '.tif'  # target extension

images = get_IDs_of_dir(input_dir_validation, ext=input_ext)  # list of input files (Validation)
targets = get_IDs_of_dir(target_dir_validation, ext=target_ext)  # list of target files (Validation)

# Pre-processing function
def preprocess(inp):
    from unet_framework.utils.general_utils import normalize01
    inp = normalize01(inp)
    inp = np.expand_dims(inp, axis=0)
    inp = np.expand_dims(inp, axis=0)
    return inp


# Post-processing function
def postprocess(inp):
    from unet_framework.utils.general_utils import re_normalize
    from unet_framework.predict.predict import _softmax
    inp = _softmax(inp)  # Softmax
    inp = inp.cpu().numpy()
    inp = inp[0]  # remove batch dimension
    inp = re_normalize(inp)  # re_normalize to range [0, 255]
    inp = np.argmax(inp, axis=0)  # Argmax
    inp = inp.astype(np.uint8)  # Type casting
    return inp


# Evaluation metrics
out_channels = 3
metrics_train_valid = {}
metrics_train_valid.update(
    {f'DSC': [metrics.DSC(index=i, argmax=False, num_classes=out_channels) for i in range(out_channels)]})
metrics_train_valid.update({f'DSC_mean': metrics.DSC(argmax=False, num_classes=out_channels)})

for sub_dir in sub_dirs:
    sub_dir_temp = os.path.join(dir, sub_dir)

    # Config file
    config_file = torch.load(os.path.join(sub_dir_temp, 'config.py'))  # Load config file
    bm = torch.load(os.path.join(sub_dir_temp, 'best_model.pt'))  # Load best model dict

    # Save root
    save_root = os.path.join(sub_dir_temp, 'predict')
    os.makedirs(save_root, exist_ok=True)  # create directory if it does not already exist

    # Load model
    model = config_file['model'].to(device, torch.float32)  # send trained model instance to device
    model.load_state_dict(bm['model'])  # Load weights

    print(f'Directory: {sub_dir_temp}')
    print(f'Score: {bm["valid_DSC_mean"]}')
    print(f'Epoch: {bm["epoch"]}')
    print('\n')

    DSC_channel = []
    DSC_mean = []

    # Stores prediction, input and target image + names
    predictions = []  # Stores every prediction
    input_images = []  # Stores every input image
    target_images = []  # Stores every target image
    input_images_names = []  # Stores every input image name

    # Prediction loop
    for i, (image_name, target_name) in enumerate(zip(images, targets)):
        pred = simple_predict(image_name, model, preprocess=preprocess, postprocess=postprocess, device=device_name)
        basename = os.path.basename(image_name).split('.')[0]  # Get basename
        input_images_names.append(basename)

        # Make directory to save files
        predict_dir = os.path.join(save_root, basename)
        os.makedirs(predict_dir, exist_ok=True)  # create directory if it does not already exist

        # Save prediction
        imsave(os.path.join(predict_dir, basename + '_pred.tif'), pred, check_contrast=False)
        predictions.append(pred)

        # Save input
        input_img = imread(image_name)
        imsave(os.path.join(predict_dir, basename + '_inp.tif'), input_img, check_contrast=False)
        input_images.append(input_img)

        # Save target
        target_img = imread(target_name)
        imsave(os.path.join(predict_dir, basename + '_tar.tif'), target_img, check_contrast=False)
        target_images.append(target_img)

        # Statistics (DSC)
        for name, evaluator in metrics_train_valid.items():
            if isinstance(evaluator, list):  # check if evaluator is a list
                DSC_channel.append([element(torch.from_numpy(target_img), torch.from_numpy(pred)) for element in evaluator])
            else:
                DSC_mean.append(evaluator(torch.from_numpy(target_img), torch.from_numpy(pred)))
        print(f'Name: {input_images_names[i]}')
        print(f'DSC_mean: {DSC_mean[i]}')
        print(f'DSC_channel: {DSC_channel[i]}')
        print('\n')

