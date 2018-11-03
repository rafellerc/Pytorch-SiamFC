from math import floor

import numpy as np
from threading import Thread
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch import sigmoid

import training.models as mdl
from appSiamFC.app_utils import get_sequence, make_gaussian_map
import utils.image_utils as imutils
from utils.tensor_conv import numpy_to_torch_var, torch_var_to_numpy

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
FLAG = 'safe'
if imutils.LIBJPEG_TURBO_PRESENT:
    FLAG = 'fast'
img_read_fcn = imutils.get_decode_jpeg_fcn(flag=FLAG)
img_resize_fcn = imutils.get_resize_fcn(flag='fast')

BufferElement = namedtuple('BufferElement', ['score_map', 'img', 'ref_img',
                                             'visible', 'name', 'bbox'])


class ProducerThread(Thread):
    """
    """

    def __init__(self, seq, buffer, dataset_path, model_path, set_type='train',
                 max_res=800, branch_arch='alexnet', ctx_mode='max'):
        """
        Args:
            seq: (int) The number of the sequence according to the get_sequence
                function, which mirrors the indexing of the ImageNetVID class.
            buffer: (queue.Queue) The data buffer between the producerThread and
                the consumer application (the display). The elements stored in
                this buffer are defined by the BufferElement namedtuple.
            dataset_path: (string) The path to the root of the ImageNet dataset.
            model_path: (string) The path to the models .pth.tar file containing
                the model's weights.
            set_type: (string) The subset of the ImageNet VID dataset, can be
                'train' or 'val'.
            max_res: (int) The maximum resolution in pixels. If any dimension
                of the image exceeds this value, the final image published by
                the producer is resized (keeping the aspect ratio). Used to
                balance the load between the consumer (main) thread and the
                producer.
            branch_arch: (string) The architecture of the branch of the siamese
                net. Might be: 'alexnet', 'vgg11_5c'.
            ctx_mode: (string) The strategy used to define the context region
                around the target, using the bounding box dimensions. The 'max'
                mode uses the biggest dimension, while the 'mean' mode uses the
                mean of the dimensions.
        """
        super(ProducerThread, self).__init__(daemon=True)
        self.frames, self.bboxes_norm, self.valid_frames, self.vid_dims = (
            get_sequence(seq, dataset_path, set_type=set_type))
        self.idx = 0
        self.seq_size = len(self.frames)
        self.buffer = buffer
        # TODO put the model info inside the checkpoint file.
        if branch_arch == 'alexnet':
            self.net = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), stride=4)
        elif branch_arch == 'vgg11_5c':
            self.net = mdl.SiameseNet(mdl.VGG11EmbeddingNet_5c(), stride=4)
        elif branch_arch == "vgg16_8c":
            self.net = mdl.SiameseNet(mdl.VGG16EmbeddingNet_8c(), stride=4)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        # Tuple of (H, w), the dimensions to which the image will be resized.
        self.resize_dims = None
        self.net = self.net.to(device)
        self.net.eval()
        self.ref, self.ref_emb = self.make_ref(ctx_mode=ctx_mode)

    @torch.no_grad()
    def run(self):
        """ The main loop of the Thread. It processes sequentially each frame
        of the specified sequence and publishes the results to the main thread
        through their shared buffer. When it finishes all the frames it sends
        a signal to the buffer indicating it is done and waits for the main
        thread to finish (because daemon=True it dies along with the main thread).
        """
        while self.idx < self.seq_size:
            dims = self.vid_dims
            if self.resize_dims is not None:
                img = img_read_fcn(self.frames[self.idx])
                img = img_resize_fcn(img, self.resize_dims, interp='bilinear')
                dims = self.resize_dims
            if self.valid_frames[self.idx]:
                bbox = self.denorm_bbox(self.bboxes_norm[self.idx], dims)
            else:
                bbox = None
            score_map = self.make_score_map(img)
            data = BufferElement(score_map,
                                 img,
                                 self.ref,
                                 self.valid_frames[self.idx],
                                 self.frames[self.idx],
                                 bbox)
            self.buffer.put(data)
            self.idx += 1
        print("ProducerThread finished publishing the data")
        # Publish a None to sinalize to the consumer that the stream has finished
        self.buffer.put(None)

    def denorm_bbox(self, bbox_norm, img_dims):
        """ Denormalizes the bounding box, taking it from its relative values to
        the pixel values in the full image with dimension img_dims.

        Args:
            bbox_norm: (list) The normalized bounding boxes, with 4 values that
                represent respectively, the x and y dimensions of the upper-left
                corner, and the width and height of the bounding boxes. All values
                are normalized by the full image's dimensions, so they are
                invariant to resizes of the image.
            img_dims: (tuple) The dimensions of the current image, in the form
                (Height, Width).
        Returns:
            bbox: (tuple) The bounding box in pixel terms, corresponding to the
                correct dimensions for an image with the given dimensions.
        """
        bbox = bbox_norm[:]
        bbox[0] = int(bbox[0]*img_dims[1])
        bbox[1] = int(bbox[1]*img_dims[0])
        bbox[2] = int(floor(bbox[2]*img_dims[1]))
        bbox[3] = int(floor(bbox[3]*img_dims[0]))
        return tuple(bbox)

    @torch.no_grad()
    def make_ref(self, ctx_mode='max'):
        """ Extracts the reference image and its embedding.

        Args:
            ctx_mode: (str) The method used to define the context region around
                the target, options are ['max', 'mean'], where 'max' simply takes
                the largest of the two dimensions of the bounding box and mean
                takes the mean.
        """
        # Get the first valid frame index
        ref_idx = self.valid_frames.index(True)
        ref_frame = img_read_fcn(self.frames[ref_idx])
        bbox = self.denorm_bbox(self.bboxes_norm[ref_idx], self.vid_dims)
        if ctx_mode == 'max':
            ctx_size = max(bbox[2], bbox[3])
        elif ctx_mode == 'mean':
            ctx_size = int((bbox[2] + bbox[3])/2)
        # It resizes the image so that the reference image has dimensions 127x127
        if ctx_size != 127:
            new_H = int(self.vid_dims[0]*127/ctx_size)
            new_W = int(self.vid_dims[1]*127/ctx_size)
            self.resize_dims = (new_H, new_W)
            ref_frame = img_resize_fcn(ref_frame, self.resize_dims, interp='bilinear')
            bbox = self.denorm_bbox(self.bboxes_norm[ref_idx], self.resize_dims)
            ctx_size = 127
        # Set image values to the range 0-1 before feeding to the network
        ref_frame = ref_frame/255
        ref_center = (int((bbox[1] + bbox[3]/2)), int((bbox[0] + bbox[2]/2)))
        ref_img = self.extract_ref(ref_frame, ref_center, ctx_size)

        ref_tensor = numpy_to_torch_var(ref_img, device)
        ref_embed = self.net.get_embedding(ref_tensor)

        return ref_img, ref_embed

    def extract_ref(self, full_img, center, ctx_size, apply_gauss=False, gauss_sig=30):
        """ Extracts the reference img from the reference frame by cropping a
        square region around the center of the bounding box with the given size.
        If the region exceeds the boundaries of the image, it pads the reference
        image with the mean value of the image.

        Args:
            full_img: (numpy.ndarray) The full reference frame.
            center: (tuple) The (y, x) coordinates of the center of the bounding
                box.
            ctx_size: (int) The side of the square region. In the current
                implementation it should be an odd integer, otherwise it would
                output an image with an excess of 1 pixel in each dimension.

        Return:
            ref_img: (numpy.ndarray) The (ctx_size, ctx_size) reference image.
        """
        H, W, _ = full_img.shape
        y_min = max(0, center[0]-ctx_size//2)
        y_max = min(H-1, center[0] + ctx_size//2)
        x_min = max(0, center[1]-ctx_size//2)
        x_max = min(W-1, center[1] + ctx_size//2)
        offset_top = max(0, ctx_size//2 - center[0])
        offset_bot = max(0, center[0] + ctx_size//2 - H + 1)
        offset_left = max(0, ctx_size//2 - center[1])
        offset_right = max(0, center[1] + ctx_size//2 - W + 1)
        img_mean = full_img.mean()
        ref_img = np.ones([ctx_size, ctx_size, 3])*img_mean
        ref_img[offset_top:(ctx_size-offset_bot),
                offset_left:(ctx_size-offset_right)] = (
                    full_img[y_min:(y_max+1), x_min:(x_max+1)])
        if apply_gauss:
            h, w, _ = ref_img.shape
            gauss = make_gaussian_map((h, w), (h//2, w//2), sig=gauss_sig)
            gauss = np.expand_dims(gauss, axis=2)
            ref_img = ref_img*gauss
        return ref_img

    def make_score_map(self, img, mode='sigmoid'):
        """
        """
        img = img/255
        # The offset is inserted so that the final size of the score map matches
        # the search image. To know more see "How to overlay the search img with
        # the score map" in Trello/Report. It is half of the dimension of the
        # Smallest Class Equivalent of the Ref image.
        offset = (((self.ref.shape[0] + 1)//4)*4 - 1)//2
        img_mean = img.mean()
        img_padded = np.pad(img, ((offset, offset), (offset, offset), (0, 0)),
                            mode='constant', constant_values=img_mean)
        img_padded = numpy_to_torch_var(img_padded, device)
        srch_emb = self.net.get_embedding(img_padded)
        score_map = self.net.match_corr(self.ref_emb, srch_emb)
        dimx = score_map.shape[-1]
        dimy = score_map.shape[-2]
        score_map = score_map.view(-1, dimy, dimx)
        if mode == 'sigmoid':
            score_map = sigmoid(score_map)
        elif mode == 'norm':
            score_map = score_map - score_map.min()
            score_map = score_map/score_map.max()
        score_map = score_map.unsqueeze(0)
        # We upscale 4 times, because the total stride of the network is 4
        score_map = F.interpolate(score_map, scale_factor=4, mode='bilinear',
                                  align_corners=False)

        score_map = score_map.cpu()
        score_map = torch_var_to_numpy(score_map)

        return score_map
