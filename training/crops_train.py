from math import floor

import numpy as np
from scipy.misc import imresize

# REMEMBER, numpy's image notation is always (Y,X,CH)


def Pads():
    """ Factory function to generate pads dictionaries with the placeholders for
    the amount of padding in each direction of an image. Note: the 'up' direction
    is up in relation to the displayed image, thus it represent the negative
    direction of the y indices.

    Return:
        pads: (dictionary) The padding in each direction.
    """
    pads = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    return pads


def crop_img(img, cy, cx, reg_s):
    """ Crops a an image given its center and the side of the crop region.
    If the crop exceeds the size of the image, it calculates how much padding
    it should have in each one of the four sides. In the case that the bounding
    box has an even number of pixels in either dimension we approach the center
    of the bounding box as the floor in each dimension, which may displace the
    actual center half a pixel in each dimension. This way we always deal with
    bounding boxes of odd size in each dimension. Furthermore, we add the same
    amount of context region in every direction, which makes the effective
    region size always an odd integer as well, so to enforce this approach we
    require the region size to be an odd integer.


    Args:
        img: (numpy.ndarray) The image to be cropped. Must be of dimensions
            [height, width, channels]
        cy: (int) The y coordinate of the center of the target.
        cx: (int) The x coordinate of the center of the target.
        reg_s: The side of the square region around the target, in pixels. Must
            be an odd integer.

    Returns:
        cropped_img: (numpy.ndarray) The cropped image.
        pads: (dictionary) A dictionary with the amount of padding in pixels in
            each side. The order is: up, down, left, right, or it can be
            accessed by their names, e.g.: pads['up']
    """
    assert reg_s % 2 != 0, "The region side must be an odd integer."
    pads = Pads()
    h, w, _ = img.shape
    context = (reg_s-1)/2  # The amount added in each direction
    xcrop_min = int(floor(cx) - context)
    xcrop_max = int(floor(cx) + context)
    ycrop_min = int(floor(cy) - context)
    ycrop_max = int(floor(cy) + context)
    # Check if any of the corners exceeds the boundaries of the image.
    if xcrop_min < 0:
        pads['left'] = -(xcrop_min)
        xcrop_min = 0
    if ycrop_min < 0:
        pads['up'] = -(ycrop_min)
        ycrop_min = 0
    if xcrop_max >= w:
        pads['right'] = xcrop_max - w + 1
        xcrop_max = w - 1
    if ycrop_max >= h:
        pads['down'] = ycrop_max - h + 1
        ycrop_max = h - 1
    cropped_img = img[ycrop_min:(ycrop_max+1), xcrop_min:(xcrop_max+1)]

    return cropped_img, pads


def resize_and_pad(cropped_img, out_sz, pads, reg_s=None, use_avg=True, resize_fcn=imresize):
    """ Resizes and pads the cropped image.

    Args:
        cropped_img: (numpy.ndarray) The cropped image.
        out_sz: (int) Output size, the desired size of the output.
        pads: (dictionary) A dictionary of the amount of pad in each side of the
            cropped image. They can be accessed by their names, e.g.: pads['up']
        reg_s: (int) Optional: The region side, used to check that the crop size
            plus the padding amount is equal to the region side in each axis.
        use_avg: (bool) Indicates the mode of padding employed. If True, the
            image is padded with the mean value, else it is padded with zeroes.

    Returns:
        out_img: (numpy.ndarray) The output image, resized and padded. Has size
            (out_sz, out_sz).
    """
    # crop height and width
    cr_h, cr_w, _ = cropped_img.shape
    if reg_s:
        assert ((cr_h+pads['up']+pads['down'] == reg_s) and
                (cr_w+pads['left']+pads['right'] == reg_s)), (
            'The informed crop dimensions and pad amounts are not consistent '
            'with the informed region side. Cropped img shape: {}, Pads: {}, '
            'Region size: {}.'
            .format(cropped_img.shape, pads, reg_s))
    # Resize ratio. Here we assume the region is always a square. Obs: The sum
    # below is equal to reg_s.
    rz_ratio = out_sz/(cr_h + pads['up'] + pads['down'])
    rz_cr_h = round(rz_ratio*cr_h)
    rz_cr_w = round(rz_ratio*cr_w)
    # Resizes the paddings as well, always guaranteeing that the sum of the
    # padding amounts with the crop sizes is equal to the output size in each
    # dimension.
    pads['up'] = round(rz_ratio*pads['up'])
    pads['down'] = out_sz - (rz_cr_h + pads['up'])
    pads['left'] = round(rz_ratio*pads['left'])
    pads['right'] = out_sz - (rz_cr_w + pads['left'])
    # Notice that this resized crop is not necessarily a square.
    rz_crop = resize_fcn(cropped_img, (rz_cr_h, rz_cr_w), interp='bilinear')
    # Differently from the paper here we are using the mean of all channels
    # not on each channel. It might be a problem, but the solution might add a lot
    # of overhead
    if use_avg:
        const = np.mean(cropped_img)
    else:
        const = 0
    # Pads only if necessary, i.e., checks if all pad amounts are zero.
    if not all(p == 0 for p in pads.values()):
        out_img = np.pad(rz_crop, ((pads['up'], pads['down']),
                                   (pads['left'], pads['right']),
                                   (0, 0)),
                         mode='constant',
                         constant_values=const)
    else:
        out_img = rz_crop
    return out_img
