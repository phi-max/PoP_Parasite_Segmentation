# %% Imports
import os

import numpy as np
import torch
from unet_framework.eval import metrics
from unet_framework.utils.general_utils import get_IDs_of_dir

# %% Directory
dir = """/data/s339697/Framework_U-Net/master_thesis/Results/UNET_2D/"""  # Specifiy the training folder here

# %% Predict validation data (Argmax)

from unet_framework.predict.predict import simple_predict
from skimage.io import imsave, imread

# Device
device_name = 'cuda'
device = torch.device(device_name)

# Config file
config_file = torch.load(os.path.join(dir, 'config.py'))  # Load config file
bm = torch.load(os.path.join(dir, 'best_model.pt'))  # Load best model dict

# Save root
save_root = os.path.join(dir, 'predict')
os.makedirs(save_root, exist_ok=True)  # create directory if it does not already exist

# Load model
model = config_file['model'].to(device, torch.float32)  # send trained model instance to device
model.load_state_dict(bm['model'])  # Load weights

print(f'Directory: {dir}')
print(f'Score: {bm["valid_DSC_mean"]}')
print(f'Epoch: {bm["epoch"]}')
print('\n')

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

DSC_channel = []
DSC_mean = []

# Stores prediction, input and target image + names
predictions = []  # Stores every prediction
input_images = []  # Stores every input image
target_images = []  # Stores every target image
input_images_names = []  # Stores every input image name

# Prediction loop
for i, (image_name, target_name) in enumerate(zip(images, targets)):
    input_img = imread(image_name)
    pred_slices = []
    for slice in input_img:
        pred = simple_predict(slice, model, preprocess=preprocess, postprocess=postprocess, device=device_name)
        pred_slices.append(pred)
    pred = np.stack(pred_slices)
    basename = os.path.basename(image_name).split('.')[0]  # Get basename
    input_images_names.append(basename)

    # Make directory to save files
    predict_dir = os.path.join(save_root, basename)
    os.makedirs(predict_dir, exist_ok=True)  # create directory if it does not already exist

    # Save prediction
    imsave(os.path.join(predict_dir, basename + '_pred.tif'), pred, check_contrast=False)
    predictions.append(pred)

    # Save input
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

# %% Load predictions validation data
from skimage.io import imread

# Get list of validation images
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

# Stores prediction, input and target image + names
predictions = []  # Stores every prediction
input_images = []  # Stores every input image
target_images = []  # Stores every target image
input_images_names = []  # Stores every input image name

# Save root
save_root = os.path.join(dir, 'predict')
if os.path.exists(save_root):
    for image_name in images:
        basename = os.path.basename(image_name).split('.')[0]
        input_images_names.append(basename)
        predict_dir = os.path.join(save_root, basename)

        # Load input
        input_img = imread(os.path.join(predict_dir, basename + '_inp.tif'))
        input_images.append(input_img)

        # Load target
        target_img = imread(os.path.join(predict_dir, basename + '_tar.tif'))
        target_images.append(target_img)

        # Load prediction
        pred = imread(os.path.join(predict_dir, basename + '_pred.tif'))
        predictions.append(pred)

        print(f'Loading: {predict_dir}')

# %% Visualize with napari
import napari
from itertools import cycle


def get_one_image_target_prediction_pair(idx):
    pred = predictions[idx]
    input_img = input_images[idx]
    target_img = target_images[idx]
    return input_img, target_img, pred


# Make a cyclable list of images
idx = list(range(len(images)))
idx = cycle(idx)

# Start napari
viewer = napari.Viewer()  # Make sure to run this in Ipython with %gui qt

# Initialize
i = next(idx)
input_img, target_img, pred = get_one_image_target_prediction_pair(i)

input_img_v = viewer.add_image(input_img, name='Input')
target_img_v = viewer.add_labels(target_img, name='Target')
pred_v = viewer.add_labels(pred, name='Prediction')
print(input_images_names[i])


# Key bindings
@viewer.bind_key('n')  # Press 'n' to show the next pair
def next_example(viewer):
    i = next(idx)
    input_img, target_img, pred = get_one_image_target_prediction_pair(i)
    print(input_images_names[i])
    input_img_v.data = input_img
    target_img_v.data = target_img
    pred_v.data = pred


# %% Statistics validation data

stats = torch.load(os.path.join(dir, 'stats.pt'))  # Load stats
training_stats = stats['training_stats']  # Load training stats
# Load best model
best_model = torch.load(os.path.join(dir, 'best_model.pt'))
best_model_epoch = best_model['epoch']
best_model_DSC_mean = best_model['valid_DSC_mean']
# Load config file
config_file = torch.load(os.path.join(dir, 'config.py'))
# Load additional statistics
stats_of_best_model = training_stats[f'{best_model_epoch}']
DSC_per_channel_best_model = stats_of_best_model['valid_DSC']
# save_root = config_file['save_root']

print('\n')
print(f'Best model')
print(f'Epoch: {best_model_epoch}')
print(f'DSC total: {best_model_DSC_mean}')
print(f'DSC per class: {DSC_per_channel_best_model[0]}')

# %% Statistics plot (Loss, learning rate, DSC mean, DSC per class)
from unet_framework.utils.general_utils import get_list_of_stats

# Load statistics
training_losses = get_list_of_stats(training_stats, 'tr_loss_mean')
validation_losses = get_list_of_stats(training_stats, 'valid_loss_mean')
learning_rate = get_list_of_stats(training_stats, 'learning_rate')
metrics_DSC_mean = get_list_of_stats(training_stats, 'valid_DSC_mean')
metrics_DSC_per_channel = get_list_of_stats(training_stats, 'valid_DSC', mode='channel')

from matplotlib import pyplot as plt
from unet_framework.utils.visual_utils import create_training_progress_plot_master

plt.style.use('seaborn-dark')  # Matplotlib style

fig = create_training_progress_plot_master(training_losses,
                                           validation_losses,
                                           learning_rate,
                                           metrics_DSC_mean,
                                           metrics_DSC_per_channel)
plt.tight_layout()
plt.show()
# %% Predict validation data (Probabilities)

from unet_framework.predict.predict import simple_predict
from skimage.io import imsave, imread

# Device
device_name = 'cuda'
device = torch.device(device_name)

# Config file
config_file = torch.load(os.path.join(dir, 'config.py'))  # Load config file
bm = torch.load(os.path.join(dir, 'best_model.pt'))  # Load best model dict

# Save root
save_root = os.path.join(dir, 'predict')
os.makedirs(save_root, exist_ok=True)  # create directory if it does not already exist

# Load model
model = config_file['model'].to(device, torch.float32)  # send trained model instance to device
model.load_state_dict(bm['model'])  # Load weights

print(f'Directory: {dir}')
print(f'Score: {bm["valid_DSC_mean"]}')
print(f'Epoch: {bm["epoch"]}')
print('\n')

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
    inp = inp.astype(np.uint8)  # Type casting
    return inp


# Stores prediction, input and target image + names
predictions = []  # Stores every prediction
input_images = []  # Stores every input image
target_images = []  # Stores every target image
input_images_names = []  # Stores every input image name

# Prediction loop
for i, (image_name, target_name) in enumerate(zip(images, targets)):
    input_img = imread(image_name)
    pred_slices = []
    for slice in input_img:
        pred = simple_predict(slice, model, preprocess=preprocess, postprocess=postprocess, device=device_name)
        pred_slices.append(pred)
    pred = np.stack(pred_slices, axis=1)
    basename = os.path.basename(image_name).split('.')[0]  # Get basename
    input_images_names.append(basename)

    predictions.append(pred)

    # Input
    input_images.append(input_img)

    # Target
    target_img = imread(target_name)
    target_images.append(target_img)

    print(f'Name: {input_images_names[i]}')
    print('\n')

# %% Visualize with napari (Probabilities)
import napari
from itertools import cycle


def get_one_image_target_prediction_pair(idx):
    pred = predictions[idx]
    input_img = input_images[idx]
    target_img = target_images[idx]
    return input_img, target_img, pred


# Make a cyclable list of images
idx = list(range(len(images)))
idx = cycle(idx)

# Start napari
viewer = napari.Viewer()  # Make sure to run this in Ipython with %gui qt

# Initialize
i = next(idx)
input_img, target_img, pred = get_one_image_target_prediction_pair(i)

input_img_v = viewer.add_image(input_img, name='Input')
target_img_v = viewer.add_labels(target_img, name='Target')
pred_v = viewer.add_image(pred, name='Prediction')
print(input_images_names[i])


# Key bindings
@viewer.bind_key('n')  # Press 'n' to show the next pair
def next_example(viewer):
    i = next(idx)
    input_img, target_img, pred = get_one_image_target_prediction_pair(i)
    print(input_images_names[i])
    input_img_v.data = input_img
    target_img_v.data = target_img
    pred_v.data = pred


# %% Save predictions in 2D (Load data first)
from unet_framework.utils.general_utils import from_grayscale_to_rgb, from_rgb_to_color
from skimage.io import imsave
from os.path import join, normpath, basename
from os import makedirs

save_dir = '/master_thesis/Images/'
save_dir = join(save_dir, basename(normpath(dir)))
makedirs(save_dir, exist_ok=True)  # create directory if it does not already exist

for input_img, target_img, pred, basename in zip(input_images, target_images, predictions, input_images_names):
    position = int(pred.shape[0] / 2)

    # Transform into RGB & color
    target_img_rgb = from_rgb_to_color(from_grayscale_to_rgb(target_img))
    pred_slice_rgb = from_rgb_to_color(from_grayscale_to_rgb(pred))

    # Take the middle slice
    input_img_slice = input_img[position]
    target_img_slice = target_img[position]
    pred_slice = pred[position]

    # Save inputs, targets, predictions
    save_dir_input = join(save_dir, basename)
    os.makedirs(save_dir_input, exist_ok=True)  # create directory if it does not already exist

    # 2D
    imsave(join(save_dir_input, basename + '_inp.tif'), input_img_slice, check_contrast=False)  # 2D
    imsave(join(save_dir_input, basename + '_tar.tif'), target_img_slice, check_contrast=False)  # 2D
    imsave(join(save_dir_input, basename + '_pred.tif'), pred_slice, check_contrast=False)  # 2D

    # 3D
    # imsave(join(save_dir_input, basename + '_inp.tif'), input_img_slice, check_contrast=False)  # 2D
    # imsave(join(save_dir_input, basename + '_tar.tif'), target_img_rgb, check_contrast=False)  # 3D
    # imsave(join(save_dir_input, basename + '_pred.tif'), pred_slice_rgb, check_contrast=False)  # 3D





# %% Predict large images (GPU, Argmax)
import unet_framework.data.Microtubules.to_predict as to_predict
from unet_framework.predict.predict import simple_predict
from skimage.io import imsave, imread

# Device
device_name = 'cuda'
device = torch.device(device_name)

# Config file
config_file = torch.load(os.path.join(dir, 'config.py'))  # Load config file
bm = torch.load(os.path.join(dir, 'best_model.pt'))  # Load best model dict

# Save root
save_root = os.path.join(dir, 'predict')
os.makedirs(save_root, exist_ok=True)  # create directory if it does not already exist

# Load model
model = config_file['model'].to(device, torch.float32)  # send trained model instance to device
model.load_state_dict(bm['model'])  # Load weights

print(f'Directory: {dir}')
print(f'Score: {bm["valid_DSC_mean"]}')
print(f'Epoch: {bm["epoch"]}')
print('\n')

images_to_predict_dir = to_predict.__path__._path[0]
img_ext = '.tif'  # input file extension
images_to_predict = get_IDs_of_dir(images_to_predict_dir, ext=img_ext)


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

# Stores prediction, input and target image + names
predictions = []  # Stores every prediction
input_images = []  # Stores every input image
input_images_names = []  # Stores every input image name

# Prediction loop
for i, image_name in enumerate(images_to_predict):
    input_img = imread(image_name)
    pred_slices = []
    for slice in input_img:
        pred = simple_predict(slice, model, preprocess=preprocess, postprocess=postprocess, device=device_name)
        pred_slices.append(pred)
    pred = np.stack(pred_slices)
    basename = os.path.basename(image_name).split('.')[0]  # Get basename
    input_images_names.append(basename)

    # Make directory to save files
    predict_dir = os.path.join(save_root, basename)
    os.makedirs(predict_dir, exist_ok=True)  # create directory if it does not already exist

    # Save prediction
    imsave(os.path.join(predict_dir, basename + '_pred.tif'), pred, check_contrast=False)
    predictions.append(pred)

    # Save input
    imsave(os.path.join(predict_dir, basename + '_inp.tif'), input_img, check_contrast=False)
    input_images.append(input_img)

    print(f'Name: {input_images_names[i]}')
    print('\n')

# %% Load predictions
from skimage.io import imread

# Get list of images
import unet_framework.data.Microtubules.to_predict as to_predict

# Load input & target
images_to_predict_dir = to_predict.__path__._path[0]
img_ext = '.tif'  # input file extension
images_to_predict = get_IDs_of_dir(images_to_predict_dir, ext=img_ext)

# Stores prediction, input and target image + names
predictions = []  # Stores every prediction
input_images = []  # Stores every input image
input_images_names = []  # Stores every input image name

# Save root
save_root = os.path.join(dir, 'predict')
if os.path.exists(save_root):
    for image_name in images_to_predict:
        basename = os.path.basename(image_name).split('.')[0]
        input_images_names.append(basename)
        predict_dir = os.path.join(save_root, basename)

        # Load input
        input_img = imread(os.path.join(predict_dir, basename + '_inp.tif'))
        input_images.append(input_img)

        # Load prediction
        pred = imread(os.path.join(predict_dir, basename + '_pred.tif'))
        predictions.append(pred)

        print(f'Loading: {predict_dir}')
# %% Visualize with napari
import napari
from itertools import cycle


def get_one_image_prediction_pair(idx):
    pred = predictions[idx]
    input_img = input_images[idx]
    return input_img, pred


# Make a cyclable list of images
idx = list(range(len(images_to_predict)))
idx = cycle(idx)

# Start napari
viewer = napari.Viewer()  # Make sure to run this in Ipython with %gui qt

# Initialize
i = next(idx)
input_img, pred = get_one_image_prediction_pair(i)

input_img_v = viewer.add_image(input_img, name='Input')
pred_v = viewer.add_labels(pred, name='Prediction')
print(input_images_names[i])


# Key bindings
@viewer.bind_key('n')  # Press 'n' to show the next pair
def next_example(viewer):
    i = next(idx)
    input_img, pred = get_one_image_prediction_pair(i)
    print(input_images_names[i])
    input_img_v.data = input_img
    pred_v.data = pred
# %% Predict large images (Probabilities)

import unet_framework.data.Microtubules.to_predict as to_predict
from unet_framework.predict.predict import simple_predict
from skimage.io import imsave, imread

# Device
device_name = 'cuda'
device = torch.device(device_name)

# Config file
config_file = torch.load(os.path.join(dir, 'config.py'))  # Load config file
bm = torch.load(os.path.join(dir, 'best_model.pt'))  # Load best model dict

# Save root
save_root = os.path.join(dir, 'predict')
os.makedirs(save_root, exist_ok=True)  # create directory if it does not already exist

# Load model
model = config_file['model'].to(device, torch.float32)  # send trained model instance to device
model.load_state_dict(bm['model'])  # Load weights

print(f'Directory: {dir}')
print(f'Score: {bm["valid_DSC_mean"]}')
print(f'Epoch: {bm["epoch"]}')
print('\n')

images_to_predict_dir = to_predict.__path__._path[0]
img_ext = '.tif'  # input file extension
images_to_predict = get_IDs_of_dir(images_to_predict_dir, ext=img_ext)


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
    inp = inp.astype(np.uint8)  # Type casting
    return inp

# Stores prediction, input and target image + names
predictions = []  # Stores every prediction
input_images = []  # Stores every input image
input_images_names = []  # Stores every input image name

# Prediction loop
for i, image_name in enumerate(images_to_predict):
    pred = simple_predict(image_name, model, preprocess=preprocess, postprocess=postprocess, device=device_name)
    basename = os.path.basename(image_name).split('.')[0]  # Get basename
    input_images_names.append(basename)

    # Prediction
    predictions.append(pred)

    # Input
    input_img = imread(image_name)
    input_images.append(input_img)

    print(f'Name: {input_images_names[i]}')
    print('\n')

# %% Visualize with napari (Probabilities)
import napari
from itertools import cycle


def get_one_image_target_prediction_pair(idx):
    pred = predictions[idx]
    input_img = input_images[idx]
    return input_img, pred


# Make a cyclable list of images
idx = list(range(len(images_to_predict)))
idx = cycle(idx)

# Start napari
viewer = napari.Viewer()  # Make sure to run this in Ipython with %gui qt

# Initialize
i = next(idx)
input_img, pred = get_one_image_target_prediction_pair(i)

input_img_v = viewer.add_image(input_img, name='Input')
pred_v = viewer.add_image(pred, name='Prediction')
print(input_images_names[i])


# Key bindings
@viewer.bind_key('n')  # Press 'n' to show the next pair
def next_example(viewer):
    i = next(idx)
    input_img, pred = get_one_image_target_prediction_pair(i)
    print(input_images_names[i])
    input_img_v.data = input_img
    pred_v.data = pred


