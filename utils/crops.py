import numpy as np
from scipy.misc import imresize


def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, use_avg=True):
    """ Pads a frame equally on all sides, enough to fit a region of size
    patch_sz centered in (pos_x, pos_y). If the region is already inside the
    frame, it doesn't do anything.

    Args:
        im: (numpy.ndarray) The image to be padded.
        frame_sz: (tuple) The width and height of the frame in pixels.
        pos_x: (int) The x coordinate of the center of the target in the frame.
        pos_y: (int) The y coordinate of the center of the target in the frame.
        path_sz: (int) The size of the patch corresponding to the context
            region around the bounding box.
        use_avg: (bool) Indicates if we should pad with the mean value of the
            pixels in the image (True) or zero (False).

    Returns:
        im_padded: (numpy.ndarray) The image after the padding.
        npad: (int) the amount of padding applied

    """
    c = patch_sz / 2
    xleft_pad = np.maximum(0, - np.round(pos_x - c))
    ytop_pad = np.maximum(0, - np.round(pos_y - c))
    xright_pad = np.maximum(0, np.round(pos_x + c) - frame_sz[1])
    ybottom_pad = np.maximum(0, np.round(pos_y + c) - frame_sz[0])
    npad = np.amax(np.asarray([xleft_pad, ytop_pad, xright_pad, ybottom_pad]))
    npad = np.int32(np.round(npad))
    paddings = ((npad, npad), (npad,npad), (0,0))
    if use_avg:
        im0 = np.pad(im[:, :, 0], paddings[0:2], mode='constant',
                     constant_values=im[:, :, 0].mean())
        im1 = np.pad(im[:, :, 1], paddings[0:2], mode='constant',
                     constant_values=im[:, :, 1].mean())
        im2 = np.pad(im[:, :, 2], paddings[0:2], mode='constant',
                     constant_values=im[:, :, 2].mean())
        im_padded = np.stack([im0,im1,im2], axis=2)
    else:
        im_padded = np.pad(im, paddings, mode='constant')
    return im_padded, npad


def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    """ Extracts the reference patch from the image.

    Args:
        im: (numpy.ndarray) The padded image.
        npad: (int) The amount of padding added to each side.
        pos_x: (int) The x coordinate of the center of the reference in the
            original frame, not considering the padding.

        pos_y: (int) The y coordinate of the center of the reference in the
            original frame, not considering the padding.
        sz_src: (int) The original size of the reference patch.
        sz_dst: (int) The final size of the patch (usually 127)
    Returns:
        crop: (numpy.ndarray) The cropped image containing the reference with its
            context region.
    """
    dist_to_side = sz_src / 2
    # get top-left corner of bbox and consider padding
    tf_x = np.int32(npad + np.round(pos_x - dist_to_side))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    tf_y = np.int32(npad + np.round(pos_y - dist_to_side))
    width = np.int32(np.round(pos_x + dist_to_side) - np.round(pos_x - dist_to_side))
    height = np.int32(np.round(pos_y + dist_to_side) - np.round(pos_y - dist_to_side))
    crop = im[tf_y:(tf_y + height), tf_x:(tf_x + width), :]
    crop = imresize(crop, (sz_dst, sz_dst), interp='bilinear')

    # TODO Add Batch dimension
    # crops = np.stack([crop, crop, crop])
    return crop


def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    """ Extracts the 3 scaled crops of the search patch from the image.

    Args:
        im: (numpy.ndarray) The padded image.
        npad: (int) The amount of padding added to each side.
        pos_x: (int) The x coordinate of the center of the bounding box in the
            original frame, not considering the padding.

        pos_y: (int) The y coordinate of the center of the bounding box in the
            original frame, not considering the padding.
        sz_src0: (int) The downscaled size of the search region.
        sz_src1: (int) The original size of the search region.
        sz_src2: (int) The upscaled size of the search region.
        sz_dst: (int) The final size for each crop (usually 255)
    Returns:
        crops: (numpy.ndarray) The 3 cropped images containing the search region
        in 3 different scales.
    """
    # take center of the biggest scaled source patch
    dist_to_side = sz_src2 / 2
    # get top-left corner of bbox and consider padding
    tf_x = npad + np.int32(np.round(pos_x - dist_to_side))
    tf_y = npad + np.int32(np.round(pos_y - dist_to_side))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = np.int32(np.round(pos_x + dist_to_side) - np.round(pos_x - dist_to_side))
    height = np.int32(np.round(pos_y + dist_to_side) - np.round(pos_y - dist_to_side))
    search_area = im[tf_y:(tf_y + height), tf_x:(tf_x + width), :]

    # TODO: Use computed width and height here?
    offset_s0 = (sz_src2 - sz_src0) / 2
    offset_s1 = (sz_src2 - sz_src1) / 2

    crop_s0 = search_area[np.int32(offset_s0):np.int32(offset_s0 + sz_src0),
                          np.int32(offset_s0):np.int32(offset_s0 + sz_src0), :]
    crop_s0 = imresize(crop_s0, (sz_dst, sz_dst), interp='bilinear')

    crop_s1 = search_area[np.int32(offset_s1):np.int32(offset_s1 + sz_src1),
                          np.int32(offset_s1):np.int32(offset_s1 + sz_src1), :]
    crop_s1 = imresize(crop_s1, (sz_dst, sz_dst), interp='bilinear')

    crop_s2 = imresize(search_area, (sz_dst, sz_dst), interp='bilinear')

    crops = np.stack([crop_s0, crop_s1, crop_s2])
    return crops
