""" This module contains the functions used to convert tensors from Pytorch to
numpy objects and vice-versa.
"""
import numpy as np
import torch


def torch_var_to_numpy(var):
    """ Converts Tensor from a Pytorch tensor to a numpy array.

    Args:
        var: (torch.Tensor) A Pytorch tensor.

    Returns:
        np_tensor: (numpy.ndarray) The same tensor as a numpy array.
    """
    np_tensor = var.detach().cpu().numpy()
    np_tensor = np.transpose(np_tensor, (0, 2, 3, 1))
    np_tensor = np_tensor.squeeze()
    return np_tensor


def numpy_to_torch_var(np_tensor, device):
    """ Converts Tensor from a numpy array to a Pytorch tensor.

    Args:
        np_tensor: (numpy.ndarray) The same tensor as a numpy array.

    Returns:
        var: (torch.Tensor) A Pytorch tensor.
    """
    if len(np_tensor.shape) == 3:
        np_tensor = np.expand_dims(np_tensor, axis=0)
    var = np.transpose(np_tensor, (0, 3, 1, 2))
    var = torch.from_numpy(var).float()
    var = var.to(device)
    return var


class ToTensorWithoutScaling(object):
    """ This function implements the ToTensor class, without scaling the pixel
    values to the [0,1] range.
    H x W x C -> C x H x W
    """

    def __call__(self, picture):
        return torch.FloatTensor(np.array(picture)).permute(2, 0, 1)