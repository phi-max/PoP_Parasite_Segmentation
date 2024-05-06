# %% Imports

import os

import numpy as np
import torch
from unet_framework.utils.general_utils import get_IDs_of_dir

# %% Directory
dir = """/data/s339697/Framework_U-Net/master_thesis/Results/UNET_Binary_CM/"""  # Specifiy the training folder here

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


