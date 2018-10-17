import numpy as np
import torch
from scipy.misc import imresize
from imageio import imread

from tracking.networks import SiamFC, load_model
from utils.crops import extract_crops_x, extract_crops_z, pad_frame
from utils.tensor_conv import numpy_to_torch_var, torch_var_to_numpy


class TrackingGraph(object):
    """ Class that defines the computations executed by the tracker.

    Attributes:
        cuda (Bool): Indicates whether to use the gpu or not.
        final_score_sz (int): The final size of the score map.
        design (named tuple): A structure holding the design specifications.
        env (named tuple): A structure holding the environment informations.
        scale_factors (ndarray): An array containing all the scales with.
        which the search image will be rescaled.
        siamese (torch.model): The torch model of SiamFC.
    """

    def __init__(self, design, env, hp, gpu=True):
        # self.pos_x = None
        # self.pos_y = None
        # self.z_sz = None
        # TODO calculate final score size based on the score size and upscale
        if (torch.cuda.is_available() and gpu):
            print('Model set to GPU')
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        self.design = design
        self.env = env
        self.hp = hp
        # Scales down and up by scale step. The center is the original scale.
        self.scale_factors = hp.scale_step ** np.linspace(-(hp.scale_num//2),
                                                          hp.scale_num//2,
                                                          num=hp.scale_num)
        self.siamese = SiamFC()
        self.siamese = load_model(self.siamese, design.net)
        self.siamese.eval()
        self.siamese.to(self.device)

    def preprocess_z(self, filename, pos_x, pos_y, z_sz):
        """ Pads and crops the examplar image based on the last tracked position

        Args:
            filename: The full path of the file containing the current frame
            pos_x: Last tracked x coordinate
            pos_y: Last tracked y coordinate
            z_sz: Last tracked context region size for the reference image

        Returns:
            z_crop: A padded and cropped version of the image, corresponding to
            the context region for the reference image.
        """
        # TODO Return torch tensor?
        image = imread(filename).astype(np.float32)
        frame_sz = image.shape[0:2]

        frame_padded_z, npad_z = pad_frame(image, frame_sz,
                                           pos_x, pos_y, z_sz,
                                           use_avg=self.design.pad_with_image_mean)
        z_crop = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz,
                                 self.design.reference_sz)
        return z_crop

    def get_template_z(self, crop_z):
        """ Gets the template for the reference crop

        Args:
            crop_z: Padded and cropped reference image

        Returns:
           The embedded template for the given crop, calculated by the network
        """
        crop_z = numpy_to_torch_var(crop_z, self.device)
        template = self.siamese.get_template(crop_z)
        template = torch_var_to_numpy(template, self.device)
        return template

    def preprocess_x(self, filename, pos_x, pos_y, z_sz):
        """ Pads and crops the search image based on the last tracked position

        Args:
            filename: The full path of the file containing the current frame
            pos_x: Last tracked x coordinate
            pos_y: Last tracked y coordinate
            z_sz: Last tracked context region size for the reference image

        Returns:
            x_crops: The stacked crops of the search image, corresponding to
            the original scale and the rescaled versions, with the appropriate
            context region of the search image.
        """
        # TODO add the option to not scale 3 times the search image
        image = imread(filename)
        image = np.float32(image)
        # TODO Check if is in range 0 - 255
        # TODO Check if the dimensions are in the right order
        frame_sz = image.shape[0:2]

        x_sz = float(self.design.search_sz)/self.design.reference_sz * z_sz

        x_sz0 = x_sz * self.scale_factors[0]
        x_sz1 = x_sz * self.scale_factors[1]
        x_sz2 = x_sz * self.scale_factors[2]

        frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x, pos_y,
                                           x_sz2, use_avg=self.design.pad_with_image_mean)
        # extracts tensor of x_crops (3 scales)
        # TODO Extend this part to an arbitrary number of scales
        if self.hp.scale_num == 3:
            x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y,
                                      x_sz0, x_sz1, x_sz2,
                                      self.design.search_sz)
        else:
            raise NotImplementedError('The extract_crops_x function hasnt yet '
                                      'been extended to arbitrary number '
                                      'of scales.')
        return x_crops

    def get_score_map(self, template_z, crops_x):
        """ Combines reference template and a search image crop to get score map

        Args:
            template_z: The embedding calculated for the reference image
            crops_x: The stacked crops of the search image

        Returns:
            score_map: The score map indicating the likelihood of the reference
            image being found centered on each pixel. The resolution of the
            score map is smaller than that of the original image, so it is
            upscaled before being returned.
        """
        # TODO give the user the choice of the number of scales (use parameters)
        template_z = np.stack([template_z]*self.hp.scale_num, axis=0)
        template_z = numpy_to_torch_var(template_z, self.device)

        if len(crops_x.shape) == 4:
            crops_x = numpy_to_torch_var(crops_x, self.device)
            score_map = self.siamese(template_z, crops_x)
            score_map = torch_var_to_numpy(score_map)
            score_map = np.transpose(score_map, (1, 2, 0))
            score_map = imresize(score_map, (self.final_score_sz,
                                             self.final_score_sz),
                                 interp='bicubic')
            score_map = np.transpose(score_map, (2, 0, 1))
        return score_map
