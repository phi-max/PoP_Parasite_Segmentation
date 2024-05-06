"""Based on https://github.com/Fangyh09/pytorch-receptive-field and https://github.com/rogertrullo/Receptive-Field-in-Pytorch"""

# %% FOV of U-Net (simple)
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def compute_N(out, f, s):
    return s * (out - 1) + f if s > 0.5 else ((out + (f - 2)) / 2) + 1


def compute_RF(layers):
    d = []
    out = 1
    for f, s in layers:
        out = compute_N(out, f, s)
        d.append(out)
    return d


depth = 4


def unet_layers(depth):
    block = [(3, 1), (3, 1), (2, 2)]
    bottleneck = [(3, 1), (3, 1)]

    n_blocks = depth - 1
    layers = (n_blocks * block) + bottleneck

    return layers


layers = unet_layers(depth=depth)

fov = compute_RF(layers)
print(f'FOV: {fov}')

# %% FOV of U-Net (fancy)

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def check_same(stride):
    if isinstance(stride, (list, tuple)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    return stride


def receptive_field(model, input_size, batch_size=-1, device="cuda"):
    '''
    :parameter
    'input_size': tuple of (Channel, Height, Width)
    :return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
    'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
        do not overlap in one direction.
        i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
    'r' for "receptive_field" is the spatial range of the receptive field in one direction.
    'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
        Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
    '''

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(receptive_field)
            m_key = "%i" % module_idx
            p_key = "%i" % (module_idx - 1)
            receptive_field[m_key] = OrderedDict()

            if not receptive_field["0"]["conv_stage"]:
                print("Enter in deconv_stage")
                receptive_field[m_key]["j"] = 0
                receptive_field[m_key]["r"] = 0
                receptive_field[m_key]["start"] = 0
            else:
                p_j = receptive_field[p_key]["j"]
                p_r = receptive_field[p_key]["r"]
                p_start = receptive_field[p_key]["start"]

                if class_name == "Conv2d" or class_name == "MaxPool2d":
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    kernel_size, stride, padding = map(check_same, [kernel_size, stride, padding])
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + (kernel_size - 1) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                elif class_name == "BatchNorm2d" or class_name == "ReLU" or class_name == "Bottleneck":
                    receptive_field[m_key]["j"] = p_j
                    receptive_field[m_key]["r"] = p_r
                    receptive_field[m_key]["start"] = p_start
                elif class_name == "ConvTranspose2d":
                    receptive_field["0"]["conv_stage"] = False
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    raise ValueError("module not ok")
                    pass
            receptive_field[m_key]["input_shape"] = list(input[0].size())  # only one
            receptive_field[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # list/tuple
                receptive_field[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                # tensor
                receptive_field[m_key]["output_shape"] = list(output.size())
                receptive_field[m_key]["output_shape"][0] = batch_size

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # create properties
    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["j"] = 1.0
    receptive_field["0"]["r"] = 1.0
    receptive_field["0"]["start"] = 0.5
    receptive_field["0"]["conv_stage"] = True
    receptive_field["0"]["output_shape"] = list(x.size())
    receptive_field["0"]["output_shape"][0] = batch_size
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format("Layer (type)", "map size", "start", "jump",
                                                             "receptive_field")
    print(line_new)
    print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4
        line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
            "",
            layer,
            str(receptive_field[layer]["output_shape"][2:]),
            str(receptive_field[layer]["start"]),
            str(receptive_field[layer]["j"]),
            format(str(receptive_field[layer]["r"]))
        )
        print(line_new)

    print("==============================================================================")
    # add input_shape
    receptive_field["input_size"] = input_size
    return receptive_field


def compute_RF_numerical(net, img_np):
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''

    img_ = Variable(torch.from_numpy(img_np).float(), requires_grad=True)
    out_cnn = net(img_)
    out_shape = out_cnn.size()
    ndims = len(out_cnn.size())
    grad = torch.zeros(out_cnn.size())
    l_tmp = []
    for i in range(ndims):
        if i == 0 or i == 1:  # batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(int(out_shape[i] / 2))
    print(tuple(l_tmp))
    grad[tuple(l_tmp)] = 1
    out_cnn.backward(gradient=grad)
    grad_np = img_.grad[0, 0].data.numpy()
    idx_nonzeros = np.where(grad_np != 0)
    RF = [np.max(idx) - np.min(idx) + 1 for idx in idx_nonzeros]

    return RF


class CNN_UNet(nn.Module):
    def __init__(self, depth):
        # layers is a list of tuples [(f,s)]
        super(CNN_UNet, self).__init__()
        self.layers = []

        # Down
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(depth - 2):
            self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())

        # Up
        self.layers.append(nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        # self.layers.append(nn.ReLU())

        self.all_layers = nn.Sequential(*self.layers)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        out = self.all_layers(x)
        return out


model = CNN_UNet(depth=4)

rf = receptive_field(model.to(device), input_size=(1, 24, 24))
