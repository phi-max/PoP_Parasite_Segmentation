import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import os

# %% Data
sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()
tanh = torch.nn.Tanh()
linear = torch.nn.Identity()

array = torch.arange(-6, 6, 0.01)

linear_array = linear(array)
tanh_array = tanh(array)
sigmoid_array = sigmoid(array)
relu_array = relu(array)

# %% Matplotlib
dpi = 300
matplotlib.rcParams['figure.dpi'] = dpi
figsize = (8, 2)

# plt.style.use('seaborn-dark')
fig = plt.figure(figsize=figsize)
grid = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

subfig1 = fig.add_subplot(grid[0, 0])
subfig2 = fig.add_subplot(grid[0, 1])
subfig3 = fig.add_subplot(grid[0, 2])
subfig4 = fig.add_subplot(grid[0, 3])

subfigures = fig.get_axes()

subfig1.plot(array, linear_array, linestyle='solid', color='red')
subfig2.plot(array, tanh_array, linestyle='solid', color='red')
subfig3.plot(array, sigmoid_array, linestyle='solid', color='red')
subfig4.plot(array, relu_array, linestyle='solid', color='red')

linewidth = 3.0

# Plot

# ax.spines['top'].set_color('none')
for sub in subfigures:
    sub.spines['bottom'].set_position('zero')
    sub.spines['left'].set_position('zero')
    sub.spines['right'].set_color('none')
    sub.spines['top'].set_color('none')

    # sub.set_xlim(-2, 2)
    # sub.set_ylim(-2, 2)
# subfig1.spines['right'].set_color('none')

subfig1.set_title('Linear')
subfig1.set_xlim(-1.5, 1.5)
subfig1.set_ylim(-1.5, 1.5)
subfig1.set_xticks([-1, 1])
subfig1.set_yticks([-1, 1])

subfig2.set_title('Sigmoid')
subfig2.set_xlim(-2.5, 2.5)
subfig2.set_ylim(-1.5, 1.5)
subfig2.set_xticks([-2, 2])
subfig2.set_yticks([-1, 1])

subfig3.set_title('Tanh')
subfig3.set_xlim(-6.5, 6.5)
subfig3.set_ylim(-0.5, 1.5)
subfig3.set_xticks([-6, 6])
subfig3.set_yticks([1])

subfig4.set_title('ReLU')
subfig4.set_xlim(-2.5, 2.5)
subfig4.set_ylim(-0.5, 1.5)
subfig4.set_xticks([-2, 2])
subfig4.set_yticks([1])

# plt.savefig(os.path.join('/data/s339697/master_thesis/Images', 'plot_activation_functions.png'), bbox_inches='tight')
plt.show()
