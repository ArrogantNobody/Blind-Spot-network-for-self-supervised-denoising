from torch import Tensor
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x


def rotate(
    x: torch.Tensor, angle: int
) -> torch.Tensor:

    if angle == 0:
        return x
    elif angle == 90:
        x = rot_img(x, np.pi/2, dtype)
        return x
    elif angle == 180:
        x = rot_img(x, np.pi, dtype)
        return x
    elif angle == 270:
        x = rot_img(x, 3 * np.pi / 2, dtype)
        return x
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")