""" This module defines the classes and functions that write the summaries
of the training using TensorboardX.
"""
from os.path import join

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

import utils.colormaps as cm


class SummaryMaker():
    """This class is mostly a wrapper around the tensorboardX SummaryWriter,
    intended to carry the current epoch index, and implement the processing of
    the outputs for more elaborate summaries.
    """

    def __init__(self, log_dir, params, up_factor):
        """
        Args:
            log_dir: (str) The path to the folder where the summary files are
                going to be written. The summary object creates a train and a
                val folders to store the summary files.
            params: (train.utils.Params) The parameters loaded from the
                parameters.json file.
            up_factor: (int) The upscale factor that indicates how much the
                scores maps need to be upscaled to match the original scale
                (used when superposing the embeddings and score maps to the
                input images).

        Attributes:
            writer_train: (tensorboardX.writer.SummaryWriter) The tensorboardX
                writer that writes the training informations.
            writer_val: (tensorboardX.writer.SummaryWriter) The tensorboardX
                writer that writes the validation informations.
            epoch: (int) Stores the current epoch.
            ref_sz: (int) The size in pixels of the reference image.
            srch_sz: (int) The size in pixels of the search image.
            up_factor: (int) The upscale factor. See Args.

        """
        # We use two different summary writers so we can plot both curves in
        # the same plot, as suggested in https://www.quora.com/How-do-you-plot-training-and-validation-loss-on-the-same-graph-using-TensorFlow%E2%80%99s-TensorBoard
        self.writer_train = SummaryWriter(join(log_dir, 'train'))
        self.writer_val = SummaryWriter(join(log_dir, 'val'))
        self.epoch = None
        self.ref_sz = params.reference_sz
        self.srch_sz = params.search_sz
        self.up_factor = up_factor

    def add_epochwise_scalar(self, mode, tag, scalar_value):
        """ Wraps the writer.add_scalar function, using the attribute epoch as
        global_step.

        Args:
            mode: (str) Indicates wheter 'train' or 'val'
            tag: (str) The tag identifying the scalar.
            scalar_value: (scalar) The value of the scalar.
        """
        if mode == 'train':
            self.writer_train.add_scalar(tag, scalar_value, self.epoch)
        elif mode == 'val':
            self.writer_val.add_scalar(tag, scalar_value, self.epoch)

    def add_overlay(self, tag, embed, img, alpha=0.8, cmap='inferno', add_ref=None):
        """ Adds to the summary the images of the input image (ref or search)
        overlayed with the corresponding embedding or correlation map. It expect
        tensors of with dimensions [C x H x W] or [B x C x H x W] if the tensor
        has a batch dimension it takes the FIRST ELEMENT of the batch. The image
        is displayed as fusion of the input image in grayscale and the overlay
        in the chosen color_map, this fusion is controlled by the alpha factor.
        In the case of the embeddings, since there are multiple feature
        channels, we show each of them individually in a grid.
        OBS: The colors represent relative values, where the peak color corresponds
        to the maximum value in any given channel, so no direct value comparisons
        can be made between epochs, only the relative distribution of neighboring
        pixel values, (which should be enough, since we are mosly interested
        in finding the maximum of a given correlation map)

        Args:
            tag: (str) The string identifying the image in tensorboard, images
                with the same tag are grouped together with a slider, and are
                indexed by epoch.
            embed: (torch.Tensor) The tensor containing the embedding of an
                input (ref or search image) or a correlation map (the final
                output). The shape should be [B, C, H, W] or [B, H, W] for the
                case of the correlation map.
            img: (torch.Tensor) The image on top of which the embed is going
                to be overlaid. Reference image embeddings should be overlaid
                on top of reference images and search image embeddings as well
                as the correlation maps should be overlaid on top of the search
                images.
            alpha: (float) A mixing variable, it controls how much of the final
                embedding corresponds to the grayscale input image and how much
                corresponds to the overlay. Alpha = 0, means there is no
                overlay in the final image, only the input image. Conversely,
                Alpha = 1 means there is only overlay. Adjust this value so
                you can distinctly see the overlay details while still seeing
                where it is in relation to the orignal image.
            cmap: (str) The name of the colormap to be used with the overlay.
                The colormaps are defined in the colormaps.py module, but values
                include 'viridis' (greenish blue) and 'inferno' (yellowish red).
            add_ref: (torch.Tensor) Optional. An additional reference image that
                will be plotted to the side of the other images. Useful when
                plotting correlation maps, because it lets the user see both
                the search image and the reference that is used as the target.

        ``Example``
            >>> summ_maker = SummaryMaker(os.path.join(exp_dir, 'tensorboard'), params,
                                           model.upscale_factor)
            ...
            >>> embed_ref = model.get_embedding(ref_img_batch)
            >>> embed_srch = model.get_embedding(search_batch)
            >>> output_batch = model.match_corr(embed_ref, embed_srch)
            >>> batch_index = 0
            >>> summ_maker.add_overlay("Ref_image_{}".format(tbx_index), embed_ref[batch_index], ref_img_batch[batch_index], cmap='inferno')
            >>> summ_maker.add_overlay("Search_image_{}".format(tbx_index), embed_srch[batch_index], search_batch[batch_index], cmap='inferno')
            >>> summ_maker.add_overlay("Correlation_map_{}".format(tbx_index), output_batch[batch_index], search_batch[batch_index], cmap='inferno')
        """
        # TODO Add numbers in the final image to the feature channels.
        # TODO Add the color bar showing the progression of values.
        # If minibatch is given, take only the first image
        # TODO let the user select the image? Loop on all images?
        if len(embed.shape) == 4:
            embed = embed[0]
        if len(img.shape) == 4:
            img = img[0]
        # Normalize the image.
        img = img/255
        embed = cm.apply_cmap(embed, cmap=cmap)
        # Get grayscale version of image by taking the weighted average of the channels
        # as described in https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html#2.-Converting-to-Grayscale
        R,G,B = img
        img_gray = 0.21 * R + 0.72 * G + 0.07 * B
        # Get the upscaled size of the embedding, so as to take into account
        # the network's downscale caused by the stride.
        upsc_size = (embed.shape[-1] - 1) * self.up_factor + 1
        embed = F.interpolate(embed, upsc_size, mode='bilinear',
                              align_corners=False)
        # Pad the embedding with zeros to match the image dimensions. We pad
        # all 4 corners equally to keep the embedding centered.
        tot_pad = img.shape[-1] - upsc_size
        # Sanity check 1. The amount of padding must be equal on all sides, so
        # the total padding on any dimension must be an even integer.
        assert tot_pad % 2 == 0, "The embed or image dimensions are incorrect."
        pad = int(tot_pad/2)
        embed = F.pad(embed, (pad, pad, pad, pad), 'constant', 0)
        # Sanity check 2, the size of the embedding in the (H, w) dimensions
        # matches the size of the image.
        assert embed.shape[-2:] == img.shape[-2:], ("The embedding overlay "
                                                    "and image dimensions "
                                                    "do not agree.")
        final_imgs = alpha * embed + (1-alpha) * img_gray
        # The embedding_channel (or feature channel) dimension is treated like
        # a batch dimension, so the grid shows each individual embeding
        # overlayed with the input image. Plus the original image is also shown.
        # If add_ref is used the ref image is the first to be shown.
        img = img.unsqueeze(0)
        final_imgs = torch.cat((img, final_imgs))
        if add_ref is not None:
            # Pads the image if necessary
            add_ref = add_ref/255
            pad = int((img.shape[-1] - add_ref.shape[-1])//2)
            add_ref = F.pad(add_ref, (pad, pad, pad, pad), 'constant', 0)
            add_ref = add_ref.unsqueeze(0)
            final_imgs = torch.cat((add_ref, final_imgs))
        final_imgs = make_grid(final_imgs, nrow=6)
        self.writer_val.add_image(tag, final_imgs, self.epoch)
