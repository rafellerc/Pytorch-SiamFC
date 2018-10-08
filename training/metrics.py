import numpy as np
from sklearn.metrics import roc_auc_score


def center_error(output, label, upscale_factor=4):
    """ The center error is a metric that measures the displacement between the
    estimated center of the target and the 'ground-truth'. One must note that
    since the dataset is not annotated in terms of target center, there is a
    intrinsic noise in the center ground-truth.
    The center_error is calculated in terms of the displacement in the search
    image in pixels, so when no upscaling is applied to the correlation map,
    the center error uses the stride of the network to calculate it.
    Networks with different strides can be directly compared using the center
    error, but networks with different reference or search image sizes are more
    tricky to be compared, since the maximum value (and the probability
    distribution) of the center error are different for different final score
    map sizes (i.e. a smaller score map is more likely to have a smaller
    center error even by random choice).

    THIS VERSION IS IMPLEMENTED WITH NUMPY, NOT PYTORCH TENSORS

    Args:
        output: (np.ndarray) The output of the network with dimension [Bx1xHxW]
        label: (np.ndarray) The labels with dimension [BxHxWx2] (Not used, kept
            for constistency with the metric_fcn(output,label) calls)
        upscale_factor: (int) Indicates the how much we must upscale the output
            feature map to match it to the input images. When we use an upscaling
            layer the upscale_factor is 1.

    Returns:
        c_error: (int) The center displacement in pixels.
    """
    b = output.shape[0]
    s = output.shape[-1]
    out_flat = output.reshape(b, -1)
    max_idx = np.argmax(out_flat, axis=1)
    estim_center = np.stack([max_idx//s, max_idx % s], axis=1)
    dist = np.linalg.norm(estim_center - s//2, axis=1)
    c_error = dist.mean()
    c_error = c_error * upscale_factor
    return c_error


def AUC(output, label):
    """ Calculates the area under the ROC curve of the given outputs. All of the
    outputs in the neutral region of the label (see labels.py) are ignored when
    calculating the AUC.
    Why I'm using SkLearn's AUC: https://github.com/pytorch/tnt/issues/54

    Args:
        output: (np.ndarray) The output of the network with dimension [Bx1xHxW]
        label: (np.ndarray) The labels with dimension [BxHxWx2]
    """
    b = output.shape[0]
    output = output.reshape(b, -1)
    mask = label[:, :, :, 1].reshape(b, -1)
    label = label[:, :, :, 0].reshape(b, -1)
    total_auc = 0
    for i in range(b):
        total_auc += roc_auc_score(label[i], output[i], sample_weight=mask[i])
    return total_auc/b


# def center_error_normalized_by_target_size():
#     pass


# The dictionary containing all the available metrics.
METRICS = {
    'AUC': {'fcn': AUC, 'kwargs': {}},
    'center_error': {'fcn': center_error,
                     'kwargs': {'upscale_factor': 4}}
}
