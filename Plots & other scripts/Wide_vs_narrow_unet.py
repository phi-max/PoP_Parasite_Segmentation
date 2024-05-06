# %% Imports etc.
import torch
from unet_framework.unet.unet import UNet

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %% Narrow
model = UNet(in_channels=1,
             out_channels=3,
             n_blocks=3,
             start_filts=32,
             normalization='group8',
             dim=2,
             conv_mode='same',
             ).to(device)

from modelsummary import summary

sum = summary(model, torch.zeros((1, 1, 100, 100)).to(device), batch_size=1, show_input=False, show_hierarchical=False)

unet_shapes = model.get_shape(input_shape=(1, 100, 100))


# %% Wide
model = UNet(in_channels=1,
             out_channels=3,
             n_blocks=3,
             start_filts=32,
             normalization='group8',
             dim=2,
             conv_mode='valid',
             ).to(device)

from modelsummary import summary

sum = summary(model, torch.zeros((1, 1, 100, 100)).to(device), batch_size=1, show_input=False, show_hierarchical=False)

unet_shapes = model.get_shape(input_shape=(1, 100, 100))