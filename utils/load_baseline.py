""" This script creates a pytorch weight file based on the file containing the
parameters of the matconvnet implementation. It has been written with the file
baseline-conv5_e55.mat (available in the author's page
http://www.robots.ox.ac.uk/~luca/cfnet.html) in mind, and has some of its
specific parameters hard coded, such as the number of layers and the composition
of each layer.
"""
import argparse
import os

import torch
import torch.optim as optim
from scipy import io
import numpy as np

import training.models as mdl

# These dictionaries store the correspondence between the layers of the matlab
# net and the pytorch one. The parameters in the .mat file are saved with names
# corresponding to the layers, e.g.: 'br_conv1f' and 'br_conv1b' contains the
# the weights and biases of the first convolutional layer.
convs = {
    'br_conv1': 0,
    'br_conv2': 4,
    'br_conv3': 8,
    'br_conv4': 11,
    'br_conv5': 14
}
bnorms = {
    'br_bn1': 1,
    'br_bn2': 5,
    'br_bn3': 9,
    'br_bn4': 12,
    'fin_adjust_bn': 'match'
}


def _import_from_matconvnet(net_path):
    """ Imports the data from the .mat file containing the parameters of the
    network. The organization of the imported .mat file is based on the
    definition of the original matlab object, thus the data is nested in various
    dictionaries and arrays, and is not very pleasing to look at.
    The basic nesting organization of the file is:
        mat_file
        └── net
            ├── vars
            ├── params
            |   └── (useless [1 1] numpy nesting)
            |       └── (useless [1 1] numpy nesting)
            |           ├── name
            |           |    └── (useless [1] numpy nesting)
            |           |        └── The actual name
            |           └── value
            |                └── (useless [1] numpy nesting)
            |                    └── The actual value
            ├── layers
            └── meta
    Args:
        net_path: (string) The path to the .mat parameter file.

    Returns:
        params_names_list: (list) The list containing the names of all the
            networks layers.
        params_values_list: (list) The list containing the parameters of all the
            networks layers.
    """
    mat = io.loadmat(net_path)
    net_dot_mat = mat.get('net')
    # organize parameters to import
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]
    return params_names_list, params_values_list


def mat2pyt_conv(W, b):
    """ Converts the parameters of a convolutional layer from the MatConvNet
    format to PyTorch Tensors with the appropriate order. The weights of the
    convolutional layers are stored in the following order:
        MatConvNet = [H, W, in_ch, out_ch]
        PyTorch = [out_ch, in_ch, H, W]
    Moreover the biases are saved in a nested array, so the dimensions when
    reading are [n 1] instead of [n].

    Args:
        W: (numpy.ndarray) The weights of a convolutional layer.
        b: (numpy.ndarray) The biases of the same layer.

    Returns:
        W: (torch.nn.Parameter) The weights as a torch parameter.
        b: (torch.nn.Parameter) The biases as a torch parameter.

    """
    W = np.transpose(W, (3, 2, 0, 1))
    W = torch.from_numpy(W).float()
    b = torch.from_numpy(b).float()
    b = b.view(torch.numel(b))
    W = torch.nn.Parameter(W)
    b = torch.nn.Parameter(b)
    return W, b


def mat2pyt_bn(param):
    """ Converts the batchnorm parameters (one at a time) into pytorch format.

    Args:
        param: (numpy.ndarray) One of the batchnorm parameters.

    Returns:
        param: (torch.nn.Parameter) The corresponding parameter in pytorch format.
    """
    param = torch.from_numpy(param).float()
    param = torch.nn.Parameter(param)
    param = param.view(torch.numel(param))
    return param


def load_baseline(net_path, pyt_net):
    """Loads the baseline weights into a pytorch SiamFC object.

    Args:
        net_path: (str) The path to the baseline file. So far it only works for
            the baseline-conv5_e55.mat file.
        pyt_net: (torch.nn.Module) The pytorch module object of the SiamFC net,
            normally created as models.SiameseNet(models.BaselineEmbeddingNet())
    Returns:
        pyt_net: (torch.nn.Module) The network with the baseline weights loaded.
    """
    # read mat file from net_path and start TF Siamese graph from placeholders X and Z
    params_names_list, params_values_list = _import_from_matconvnet(net_path)
    for conv_name, pyt_layer in convs.items():
        conv_W_name = conv_name + 'f'
        conv_b_name = conv_name + 'b'
        conv_W = params_values_list[params_names_list.index(conv_W_name)]
        conv_b = params_values_list[params_names_list.index(conv_b_name)]
        W, b = mat2pyt_conv(conv_W, conv_b)
        # The original Network has been trained with RGB images with pixels
        # in the range 0-255, in order to use this network with our 0-1 range
        # images we need to multiply the first layer weights by 255.
        if conv_W_name == 'br_conv1f':
            W = 255 * W
            converted_weights = True
        conv_dict = {'weight': W, 'bias': b}
        pyt_net.embedding_net.fully_conv[pyt_layer].load_state_dict(conv_dict)

    # Sanity-check
    assert converted_weights, ("The weights of the first layer have not been "
                               "multiplied by 255. 0-1 range images won't work")

    for bn_name, pyt_layer in bnorms.items():
        bn_beta_name = bn_name + 'b'
        bn_gamma_name = bn_name + 'm'
        bn_moments_name = bn_name + 'x'
        bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
        bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
        bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
        bn_moving_mean = bn_moments[:, 0]
        bn_moving_variance = bn_moments[:, 1]**2  # saved as std in matconvnet
        beta = mat2pyt_bn(bn_beta)
        gamma = mat2pyt_bn(bn_gamma)
        mean = mat2pyt_bn(bn_moving_mean)
        var = mat2pyt_bn(bn_moving_variance)
        bn_dict = {'bias': beta, 'weight': gamma, 'running_mean': mean,
                   'running_var': var}
        if pyt_layer != 'match':
            pyt_net.embedding_net.fully_conv[pyt_layer].load_state_dict(bn_dict)
        else:
            pyt_net.match_batchnorm.load_state_dict(bn_dict)

    return pyt_net


def export_to_checkpoint(state, checkpoint):
    """ Saves the network as a checkpoint file in the given path.
    Args:
        state: (dict) contains model's state_dict, may contain other keys such
        as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'baseline_pretrained.pth.tar')
    torch.save(state, filepath)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Load Baseline script")
    parser.add_argument('-n', '--net_file_path',
                        default='baseline-conv5_e55.mat',
                        help="The path to the .mat file containing "
                             "the networks weights.")
    parser.add_argument('-d', '--dst_path',
                        default='training/experiments/default',
                        help="The destination path on which "
                             "the pretrained model will be saved.")
    args = parser.parse_args()
    return args


def main(args):
    """ Execute this file as a script and it will save the baseline model in
    the default experiment folder.
    """
    net_path = args.net_file_path
    net = mdl.SiameseNet(mdl.BaselineEmbeddingNet())
    net = load_baseline(net_path, net)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    export_to_checkpoint({'epoch': 0,
                          'state_dict': net.state_dict(),
                          'optim_dict': optimizer.state_dict()},
                         checkpoint=args.dst_path)


if __name__ == '__main__':
    args_ = parse_arguments()
    main(args_)
