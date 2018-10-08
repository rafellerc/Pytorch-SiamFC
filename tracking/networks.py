from os.path import abspath, join, dirname

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFC(nn.Module):
    """ Class defines the network architecture of the SiamFC, according to the
    original 5 layer, AlexNet based architecture proposed in Bertinetto's paper

    Attributes:
        branch (nn.Module): The branch of the siamese network, responsible for
        embedding the template and search image.
        match_batchnorm (nn.Module): The correlation stage of the architecture
        that correlates the search embedding with the embedded template.
    """

    def __init__(self):
        super(SiamFC, self).__init__()
        layer_list = [nn.Conv2d(3, 96, kernel_size=11, stride=2, bias=True),
                      nn.BatchNorm2d(96),
                      nn.ReLU(),
                      nn.MaxPool2d(3, stride=2),
                      nn.Conv2d(96, 256, kernel_size=5, stride=1,
                                groups=2, bias=True),
                      nn.BatchNorm2d(256),
                      nn.ReLU(),
                      nn.MaxPool2d(3, stride=1),
                      nn.Conv2d(256, 384, kernel_size=3, stride=1,
                                groups=1, bias=True),
                      nn.BatchNorm2d(384),
                      nn.ReLU(),
                      nn.Conv2d(384, 384, kernel_size=3, stride=1,
                                groups=2, bias=True),
                      nn.BatchNorm2d(384),
                      nn.ReLU(),
                      nn.Conv2d(384, 32, kernel_size=3, stride=1,
                                groups=2, bias=True)]
        self.branch = nn.ModuleList(layer_list)
        self.match_batchnorm = nn.BatchNorm2d(1)

    def get_template(self, x):
        """ Passes an image through the branch.

        Args:
            x (torch.Tensor): The image, usually a 127x127 or 255x255 image.
        Returns:
            x (torch.Tensor): The embedded tensor.
        """
        for layer in self.branch:
            x = layer(x)
        return x

    def forward(self, template_z, search_region_x):
        """ The forward pass method that takes a template and a search region.

        Args:
            template_z (torch.Tensor): The template of the reference image.
            search_region_x (torch.Tensor): An 3-color-channel crop, normally
            255x255, corresponding to the search region of the original image.
        Returns:
            match_map (torch.Tensor):
        """
        template_x = self.get_template(search_region_x)
        # The order matters in this cross-correlation layer
        match_map = F.conv2d(template_x, template_z)
        match_map = torch.sum(match_map, dim=1)  # Sums over the channel dim
        match_map = match_map.view(match_map.size()[0], 1, match_map.size()[1],
                                   match_map.size()[2])
        match_map = self.match_batchnorm(match_map)

        return match_map


def load_model(model, model_name):
    """ Loads the weights of the network.

    Args:
        model (nn.Module): The non-initialized network object, of class SiamFC
        exp_name (string): The name of the experiment folder being used.
        model_name (string): The name of the pretrained weights saved inside
        the root/models folder.
    Returns:
        model (nn.Module): The initialized network.
    """
    root_path = dirname(dirname(abspath(__file__)))
    model_path = join(root_path, 'models', model_name)

    model.load_state_dict(torch.load(model_path))
    return model
