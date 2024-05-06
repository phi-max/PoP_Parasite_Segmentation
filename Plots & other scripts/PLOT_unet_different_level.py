from unet_framework.utils.visual_utils import create_UNet_architecture
from unet_framework.utils.general_utils import calculate_output_shape_unet

shape = (1, 256, 256)
u_net1 = calculate_output_shape_unet(input_shape=shape, depth=3, filters=16, labels=3, conv_mode='same')
arch1 = create_UNet_architecture(u_net1)

u_net2 = calculate_output_shape_unet(input_shape=shape, depth=4, filters=16, labels=3, conv_mode='same')
arch2 = create_UNet_architecture(u_net2)

u_net3 = calculate_output_shape_unet(input_shape=shape, depth=5, filters=16, labels=3, conv_mode='same')
arch3 = create_UNet_architecture(u_net3)