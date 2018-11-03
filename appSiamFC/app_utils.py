import glob
import os
from os.path import join, basename, dirname

import numpy as np

from training.train_utils import get_annotations


def get_sequence(seq_num, imagenet_dir, set_type='train'):
    """
    """
    dir_data = join(imagenet_dir, 'Data', 'VID', set_type)
    dir_annot = join(imagenet_dir, 'Annotations', 'VID', set_type)
    if set_type == 'val':
        glob_expression = join(dir_data, '*')
    elif set_type == 'train':
        glob_expression = join(dir_data, '*', '*')

    sequence_path = sorted(glob.glob(glob_expression))[seq_num]
    frames = []
    bboxes_norm = []
    valid_frame = []

    for frame in sorted(os.listdir(sequence_path)):
        frame_path = join(dir_annot, sequence_path, frame)
        # Get the relative path to the sequence form the annotations directory
        if set_type == 'val':
            sequence_rel_path = basename(sequence_path)
        elif set_type == 'train':
            # Yes, I'm aware this line is a monstrosity. 
            sequence_rel_path = join(basename(dirname(sequence_path)), basename(sequence_path))
        annot, w, h, valid = get_annotations(dir_annot, sequence_rel_path, frame)
        # The bounding box is written in terms of the x and y coordinate of the
        # top left corner in proportional terms of the sides of the whole picture.
        # So to get the pixel values for it one must multiply the x dimensions
        # 0 and 2 by the width of the image and the y dimensions by the height.
        # We use this notation because it is invariant to any scaling of the
        # image.
        if valid:
            bbox_norm = [annot['xmin']/w,
                         annot['ymin']/h,
                         (annot['xmax']-annot['xmin'])/w,
                         (annot['ymax']-annot['ymin'])/h]
        else:
            bbox_norm = None
        frames.append(frame_path)
        bboxes_norm.append(bbox_norm)
        valid_frame.append(valid)
        dims = (h, w)

    return frames, bboxes_norm, valid_frame, dims


def make_gaussian_map(dims, center, sig):
    """
    """
    var = sig**2
    x = np.arange(dims[1])[None].astype(np.float)
    y = np.arange(dims[0])[None].astype(np.float).T
    gauss = np.exp(-(y-center[0])**2/(2*var))*(np.exp(-(x-center[1])**2/(2*var)))

    return gauss


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.stack([gray, gray, gray], axis=2)

    return gray