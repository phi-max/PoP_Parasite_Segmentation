import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import os

# %% Data
relu = torch.nn.ReLU()
elu = torch.nn.ELU()

array = torch.arange(-6, 6, 0.01)

relu_array = relu(array)
elu_array = elu(array)

# %% Matplotlib
dpi = 300
matplotlib.rcParams['figure.dpi'] = dpi
figsize = (8, 3)

# plt.style.use('seaborn-dark')
fig = plt.figure(figsize=figsize)
grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

subfig1 = fig.add_subplot(grid[0, 0])
subfig2 = fig.add_subplot(grid[0, 1])

subfigures = fig.get_axes()

subfig1.plot(array, relu_array, linestyle='solid', color='red')
subfig2.plot(array, elu_array, linestyle='solid', color='red')

linewidth = 3.0

# Plot

# ax.spines['top'].set_color('none')
for sub in subfigures:
    sub.spines['bottom'].set_position('zero')
    sub.spines['left'].set_position('zero')
    sub.spines['right'].set_color('none')
    sub.spines['top'].set_color('none')

subfig1.set_title('ReLU')
subfig1.set_xlim(-3.5, 2.5)
subfig1.set_ylim(-1.5, 1.5)
subfig1.set_xticks([-2, 2])
subfig1.set_yticks([1])

subfig2.set_title('ELU')
subfig2.set_xlim(-3.5, 2.5)
subfig2.set_ylim(-1.5, 1.5)
subfig2.set_xticks([-2, 2])
subfig2.set_yticks([1])

# plt.savefig(os.path.join('/data/s339697/master_thesis/Images', 'ReLU_vs_ELU.png'), bbox_inches='tight')
plt.show()
