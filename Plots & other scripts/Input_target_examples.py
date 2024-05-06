import os
from skimage.io import imread
import matplotlib
from matplotlib import pyplot as plt

from unet_framework.utils.general_utils import from_grayscale_to_rgb, from_rgb_to_color, get_IDs_of_dir

matplotlib.rcParams['figure.dpi'] = 300

position = 11

# %% Config file
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

img1 = imread(train_input_IDs[0])
tar1 = from_rgb_to_color(from_grayscale_to_rgb(imread(train_target_IDs[0])))

img2 = imread(train_input_IDs[1])
tar2 = from_rgb_to_color(from_grayscale_to_rgb(imread(train_target_IDs[1])))

from skimage.io import imsave

imsave(os.path.join('/data/s339697/master_thesis/Images', 'example_input_01.png'), img1[position])
imsave(os.path.join('/data/s339697/master_thesis/Images', 'example_input_02.png'), img2[position])

imsave(os.path.join('/data/s339697/master_thesis/Images', 'example_target_01.png'), tar1[position])
imsave(os.path.join('/data/s339697/master_thesis/Images', 'example_target_02.png'), tar2[position])



imgs = [img1, img2]
tars = [tar1, tar2]

# Plot
fig, fig_axes = plt.subplots(ncols=2, nrows=2, figsize=(4, 4), gridspec_kw={'wspace': 0.005, 'hspace': 0.005})

fontdict = {'fontsize': 14}

for x_pos, row in enumerate(fig_axes):
    for y_pos, subfig in enumerate(row):
        # subfig.axis('off')
        subfig.tick_params(axis=u'both', which=u'both', length=0)
        subfig.set_xticklabels([])
        subfig.set_yticklabels([])

for i, subfig in enumerate(fig_axes[:, 0]):
    subfig.imshow(imgs[i][position], cmap='gray')

for i, subfig in enumerate(fig_axes[:, 1]):
    subfig.imshow(tars[i][position], cmap='gray')

for i, subfig in enumerate(fig_axes[0]):
    if i == 0:
        subfig.set_title('Input', fontdict=fontdict)
    if i == 1:
        subfig.set_title('Target', fontdict=fontdict)

plt.style.use('seaborn-dark')
# plt.savefig(os.path.join('/home/s339697/Documents/', 'Labels.png'), bbox_inches='tight')
plt.show()
