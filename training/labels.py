import numpy as np


def create_BCELogit_loss_label(label_size, pos_thr, neg_thr, upscale_factor=4):
    """ Creates the label for the output of a training pair. Because both crops
    are always centered, the label doesn't depend on the particular positions
    of the target on their respective images.
    This label has a equivalent effect than logistic_loss_label, when used with
    the Binary Cross Entropy Logit loss, but saves processing time and is more
    in line with common ML practices. To model the behavior of the 3-level
    labels of the logistic loss it has two dimensions for each pixel, one that
    encodes the positive (1) and negative (0) pixels and a second one that
    encodes if the pixel is positive/negative (1) or neutral (0). The neutral
    values do not contribute to the gradient, so they are simply ignored when
    calculating the loss, and their value in the first dimension can be set
    arbitrarily.
    The thresholds are defined in terms of pixel distances in the search image,
    so, if the correlation map is not upscaled (i.e. it has a stride > 1) then
    we must convert the thresholds to the corresponding distances in the
    correlation map (by simply dividing the thresholds by the stride).
    The correlation map corresponds to a centered subregion of the search image
    (typically, for a search image of 255 pixels, and ref image of 127, this
    subregion has 129 pixels) so there are practical limits to the thresholds:
    any value larger than the half diagonal of this subregion (typically, 91.2
    pixels) will encompass the whole correlation map.

    The tests are dist <= pos_thr and dist > neg_thr.

    Args:
        label_size: (int) The size in pixels of the output of the network.
        pos_thr: (int) The positive values threshold, in terms of the pixels of
            the search image.
        neg_thr: (int) The negative values threshold, in terms of the pixels of
            the search image.
        upscale_factor: (int) How much we have to upscale the output feature map
            to match the input images. It represents the mean displacement of
            the receptive field of the network in the input images for a
            one-pixel movement in the correlation map. Typically it tells how
            much the network up or downscales the image.

    Returns:
        label (numpy.ndarray): The label for each pair with dimension
            label_size * label_size * 2

    OBS: Set both pos_thr and neg_thr to zero to get the label for an empty
        frame
    """
    # Convert the thresholds if the stride is bigger than 1, else it stays the same.
    pos_thr = pos_thr / upscale_factor
    neg_thr = neg_thr / upscale_factor
    # Note that the center might be between pixels if the label size is even.
    center = (label_size - 1) / 2
    line = np.arange(0, label_size)
    line = line - center
    line = line**2
    line = np.expand_dims(line, axis=0)
    dist_map = line + line.transpose()

    label = np.zeros([label_size, label_size, 2]).astype(np.float32)
    label[:, :, 0] = dist_map <= pos_thr**2
    label[:, :, 1] = (dist_map <= pos_thr**2) | (dist_map > neg_thr**2)

    return label
