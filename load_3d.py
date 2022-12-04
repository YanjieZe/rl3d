import torch
from termcolor import colored
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

def visual_representation(ckpt_path=None, use_3d=False):
    """
    2D encoder of pretrained 3D Visual representation
    """
    if ckpt_path is None:
        ckpt_path = "checkpoints/videoae_co3d.tar"
    # check path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Checkpoint path does not exist")
    encoder_3d = nn.DataParallel(Encoder3D())
    checkpoint = torch.load(ckpt_path)
    encoder_3d.load_state_dict(checkpoint['encoder_3d'])
    print(colored(">>pretrained 3D visual representation is loaded.", "red"))
    if use_3d:
        return encoder_3d
    else:
        encoder_2d = encoder_3d.module.feature_extraction
        return encoder_2d


def get_resnet18():
    model = torchvision.models.resnet18(pretrained=True)
    feature = nn.Sequential(*list(model.children())[:-2])
    feature[7][0].conv1.stride = (1, 1)
    feature[7][0].downsample[0].stride = (1, 1)
    return feature

class Encoder3D(nn.Module):
    """
    Encoder 3D v2.
    """

    def __init__(self, args=None):
        super(Encoder3D, self).__init__()
        self.backbone = "resnet18"
        if self.backbone=="resnet18":
            self.feature_extraction = get_resnet18()
            self.conv3d_1 = nn.ConvTranspose3d(64, 48, 4, stride=2, padding=1)
            self.conv3d_2 = nn.ConvTranspose3d(48, 32, 4, stride=2, padding=1)
        else:
            raise NotImplementedError

    def forward(self, img, use_3d=False):
        z_2d = self.feature_extraction(img)
        if use_3d:
            B,C,H,W = z_2d.shape
            if self.backbone=="resnet18":
                z_3d = z_2d.reshape([-1, 64, 8, H, W])
            elif self.backbone=="resnet50":
                z_3d = z_2d.reshape([-1, 256, 8, H, W])
            z_3d = F.leaky_relu(self.conv3d_1(z_3d))
            z_3d = F.leaky_relu(self.conv3d_2(z_3d))
            return z_3d
        else:
            return z_2d



