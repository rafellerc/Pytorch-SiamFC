""" This file defines the functions that convert tensors of 2D single channel
channel images (2D tensors, or some collection of 2D tensors) into a collection
of colored 2D RGB maps (3D tensors), for better visualization. Its main use case
is to transform a score map, where the score is a 1 dimensional number for each
pixel of the image, into a colored image.
It defines the values of such mappings, based on https://bids.github.io/colormap/
which defines colorblind-friendly 1D -> [RGB] mappings. This file was written
to be used with tensorboardX, thus only uses pytorch operations, and can be run
on a gpu.
"""
from .color_tables import VIRIDIS_cm as VIRIDIS
from .color_tables import INFERNO_cm as INFERNO


def apply_cmap(tensor, dim=-3, cmap='inferno', norm='per_channel'):
    """ Applies a colormap to a collection of 2D score maps. The tensor of score
    maps might be a single score map of shape [H, W], a collection of score maps
    with multiple feature channels of shape [C, H, W] or a batch of collections
    of shape [B, C, H, W]. The transformation consists of mapping the values
    of each pixel to 256 different RGB values defined in one of COLOR_MAPS
    color maps. The available colormaps are listed in the COLOR_MAPS dictionary.
    The mapping takes into account the range of the data inside the tensor,
    if you set the normalization to 'per_channel' t

    Args:
        tensor: (torch.Tensor) The tensor containing the score maps, can have
            different shapes, but the two last dimensions must be H and W.
        dim: (int) The dimension of the final tensor where you wish to put
            the RGB channels, by default it is the third dimension from the end,
            making the final tensor something of shape [..., RGB_Ch, H, W]. It
            supports python's negative indexing.
        cmap: (str) The name of the particular colormap to be used. Default
            value is 'inferno', which is a blueish black to yellowish white
            colormap.
        norm: (str) The type of normalization to be used. 'per_channel'
            indicates that the min and max values are calculated in each individual
            channel, while 'per_batch' calculates the min and max values per batch.

    Returns:
        tensor: (torch.Tensor) The tensor containing the input score maps,
            converted to colored maps based on their pixel values. It adds
            one dimension of size 3 to the tensor (the 3 RGB channels), the
            position of such dimension is defined by the user, but by default it
            is the third to last dimension.
    """
    # Normalize
    if norm == 'per_channel' or len(tensor.shape) == 2:
        # Performs channel-wise normalization
        # First we have to squash the H and W dimensions to get the min and max
        # in each 2D map. Then we subtract the min of each channel, and then
        # we divide by the max of each channel (of the subtracted tensor).
        *v_dims, H, W = tensor.shape
        tensor = tensor - tensor.view(*v_dims, -1).min(-1)[0].view(*v_dims, 1, 1)
        tensor = tensor/tensor.view(*v_dims, -1).max(-1)[0].view(*v_dims, 1, 1)
    elif norm == 'per_batch':
        # Performs batch-wise normalization
        *v_dims, C, H, W = tensor.shape
        tensor = tensor - tensor.view(*v_dims, -1).min(-1)[0].view(*v_dims, 1, 1, 1)
        tensor = tensor/tensor.view(*v_dims, -1).max(-1)[0].view(*v_dims, 1, 1, 1)
    # Converts all the values to 0-255 integers to be used as indexes to the color map
    tensor = 255*tensor
    tensor.floor_()
    tensor = tensor.long()
    # Get the [256, 3] color map tensor
    color_map = COLOR_MAPS[cmap]
    if tensor.is_cuda:
        color_map = color_map.cuda()
    # Transform the tensor by using its values as indexes to the color map.
    # This constructions might seem confusing, but essentially it accesses each
    # element of tensor and uses its value as index to the color_map, returning
    # a 3D vector (a RGB pixel) for each element of tensor. Thus the right side
    # tensor has a size 3 dimension added to the original tensor shape.
    tensor = color_map[tensor]
    # The RGB channels dimension is added in the end of the tensor, so we must
    # permute the dimensions to put the RGB dimension in the requested position.

    # First off, in case the user uses a negative indexing we convert it to
    # the positive equivalent
    if dim < 0:
        dim += len(tensor.shape)
    # Then we generate a list of the original dimensions
    permutation = list(range(len(tensor.shape)-1))
    # And insert the last dimension (the RGB dim) in the requested position
    permutation.insert(dim, -1)
    # And we permute the tensors dimensions using this list. Note that the non-
    # RGB dimensions are left unchanged.
    tensor = tensor.permute(permutation)
    return tensor


COLOR_MAPS = {'viridis': VIRIDIS, 'inferno': INFERNO}
