import torch
from Unet.model import Unet as _UNet

def unet_f(pretrained=False, scale=0.5):
 
    net = _UNet(n_channels=3, n_classes=2, bilinear=False)
    if pretrained:
        if scale == 0.5:
            checkpoint = 'https://github.com/Markinllz/Unet-implemenation-for-all-datasets-/blob/main/data/Data/checkpoints/checkpoint_epoch5.pth'
        elif scale == 1.0:
            checkpoint = 'https://github.com/Markinllz/Unet-implemenation-for-all-datasets-/blob/main/data/Data/checkpoints/checkpoint_epoch5.pth'
        else:
            raise RuntimeError('Only 0.5 and 1.0 scales are available')
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')
        net.load_state_dict(state_dict)

    return net