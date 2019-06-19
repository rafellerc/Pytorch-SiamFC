import os
from os.path import join, relpath, isfile
from math import sqrt
import random
import glob
import json

import numpy as np
from imageio import imread
from scipy.misc import imresize
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from training.crops_train import crop_img, resize_and_pad
from utils.exceptions import IncompatibleImagenetStructure
from training.train_utils import get_annotations, check_folder_tree
from training.labels import create_BCELogit_loss_label as BCELoss


class ImageNetVID(Dataset):
    """ Dataset subclass representing the ImageNet VID train dataset. It was
    designed to work with the out-of-the box folder structure of the dataset,
    so the folder organization should have the following structure:
    Imagenet/
    ├── Annotations
    │   └── VID
    │       ├── train
    │       └── val
    └── Data
        └── VID
            ├── train
            └── val
    The other folders present in the ImageNet VID (ImageSets, test, snippets)
    are ignored and can be safely removed if needed.
    OBS: the validation dataset inherits from this class.
    """

    def __init__(self, imagenet_dir, transforms=ToTensor(),
                 reference_size=127, search_size=255, final_size=33,
                 label_fcn=BCELoss, upscale_factor=4,
                 max_frame_sep=50, pos_thr=25, neg_thr=50,
                 cxt_margin=0.5, single_label=True, img_read_fcn=imread,
                 resize_fcn=imresize, metadata_file=None, save_metadata=None):
        """ Initializes the ImageNet VID dataset object.

        Args:
            imagenet_dir: (str) The full path to the ImageNet VID dataset root
                folder.
            transforms: (torchvision.transforms.transforms) A composition of
                transformations using torchvision.transforms.Compose (or a
                single transformation) to be applied to the data.
            reference_size: The total size of the side of the reference region,
                in pixels.
            search_size: The total size of the side of the search region,
                in pixels.
            final_size: The size of the output of the network.
            label_fcn: The function that generates the labels for the output of
                the network. Should match your definition of loss function.
            upscale_factor: The mean displacement of the receptive field in the
                input images as we move the output of one pixel. Typically, it
                is equal to the sum of the network's strides when there is no
                upscaling, and 1 when there is a full upscaling.
            max_frame_sep: The reference/search pairs are chosen (pseudo)-randomly
                from frames that are separated by at most 'max_frame_sep' frames.
                This parameter influences the capacity of the model to represent
                long-term model appearance changes.
            single_label: (bool) Indicates if we are always using the same label,
                so that we might create it once and for all, instead of calling
                the label function to create the same label at each iteration.
            img_read_fcn: (function) The function used to read the images, by
                default it is the Scikit-image imread function.
            resize_fcn: (function) The function used to resize the images of the
                dataset. Default function is scipy's imresize.
            metadata_file: (str) The path to the file containing the
                .metadata file to be loaded.
            save_metadata: (str) The path to the file where you want to
                save the new .metadata file in the case that no metadata file
                could be loaded. Set to none if you don't want to save the metadata.
        """
        # Do a basic check on the structure of the dataset file-tree
        if not check_folder_tree(imagenet_dir):
            raise IncompatibleImagenetStructure
        self.set_root_dirs(imagenet_dir)
        self.max_frame_sep = max_frame_sep
        self.reference_size = reference_size
        self.search_size = search_size
        self.upscale_factor = upscale_factor
        self.cxt_margin = cxt_margin
        self.final_size = final_size
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.transforms = transforms
        self.label_fcn = label_fcn
        if single_label:
            self.label = self.label_fcn(self.final_size, self.pos_thr,
                                        self.neg_thr,
                                        upscale_factor=self.upscale_factor)
        else:
            self.label = None
        self.img_read = img_read_fcn
        self.resize_fcn = resize_fcn
        # Create Metadata. This section is specific to the train version
        self.get_metadata(metadata_file, save_metadata)

    def set_root_dirs(self, root):
        self.dir_data = join(root, 'Data', 'VID', "train")
        self.dir_annot = join(root, 'Annotations', 'VID', "train")

    def get_scenes_dirs(self):
        glob_expression = join(self.dir_data, '*', '*')
        relative_paths = [relpath(p, self.dir_data) for p in sorted(glob.glob(glob_expression))]
        return relative_paths

    def get_metadata(self, metadata_file, save_metadata):
        """ Gets the metadata, either by loading it from a json file or by
        building it from scratch. If the metadata is built, we might save it
        for later use as a json metadata file.

        Args:
            metadata_dir: (str or None) The path to the metadata file.
                Set to None to build the metadata from scratch, based on the
                imagenet directory attribute of the object.
            save_metadata: (str or None) The path of the file you want the
                metadata to be saved on in the case the program has to build it
                from scratch. If None, the program will not save the metadata
                in any case.
                Note: even if save_metadata specifies a valid path, no metadata
                files will be saved if the program is able to get the metadata
                from metadata file.
        """
        # Check if not None
        if metadata_file and isfile(metadata_file):
            with open(metadata_file) as json_file:
                mdata = json.load(json_file)
            # Check the metadata file.
            if self.check_metadata(mdata):
                print("Metadata file found. Loading its content.")
                for key, value in mdata.items():
                    setattr(self, key, value)
                return
        # If couldn't get metadata from file, build it from scratch
        mdata = self.build_metadata()
        if save_metadata is not None:
            with open(save_metadata, 'w') as outfile:
                json.dump(mdata, outfile)

    def check_metadata(self, metadata):
        """ Checks if the metadata loaded from file is correct. Doesn't make
        a thorough verification, the main case we are trying to avoid is where
        the metadata was built on a different machine or the data have been
        moved around.

        Args:
            metadata: (dictionary) The dictionary loaded from the json metadata
                file. It shoud contain the keys 'frames', 'annotations' and
                'list_idx'.
        Returns:
            Bool: Indicates whether the dictionary corresponds to our expectations,
                if not it simply returns False, and the program should build
                the metadata from scratch.
        """
        if not all(key in metadata for key in ('frames', 'annotations', 'list_idx')):
            return False
        # Check if first and last frames exist in indicated paths.
        if not (isfile(metadata['frames'][0][0]) and isfile(metadata['frames'][-1][-1])):
            return False
        return True

    def get_seq_name(self, seq_idx):
        """ Returns the name of the video sequence corresponding to seq_idx, which
        allows the user to identify the video in the ImageNet folder.

        Args:
            seq_idx: (int) The index of the sequence according to self.frames
        Returns:
            The name of the sequence's folder. E.g. 'ILSVRC2015_val_00007029'
        """
        return self.frames[seq_idx][0].split(os.sep)[-2]

    def build_metadata(self):
        """ Builds the metadata from scratch. Gets the
        paths to each of the frames as well as their annotations.

        Returns:
            frames: (list) A list with all the paths to the frames in the
                concerned set. Organized as a list of lists, where the first
                dimension indicates the sequence and the second, the frame number
                inside that sequence.
            annotations: (list) A list of list of dictionaries containing the
                annotations for each frame. The list is organized the same way
                as frames, so frames[seq][frame] and annotations[seq][frame]
                correspond to the same frame. The annotation information can be
                accessed using the keywords 'xmax', 'xmin', 'ymax', 'ymin':
                Example: del_x =  annotation['xmax'] - annotation['xmin']
            list_idx: (list) A flat list with lenght equal to the total number
                of valid frames in the set. Each element is the sequence number
                of the corresponding frame, so that when we're using the
                __getitem__ method, we can know the sequence of a frame of index
                idx as list_idx[idx]. For example, if the first sequence has 3
                frames and the second one has 4, the list would be:
                [0,0,0,1,1,1,1,2,...]
        """
        frames = []
        annotations = []
        list_idx = []

        # In the training folder the tree depth is one level deeper, so that
        # a sequence folder in training would be:
        # 'ILSVRC2015_VID_train_0000/ILSVRC2015_train_00030000', while in
        # the val set it would be: 'ILSVRC2015_val_00000000'
        scenes_dirs = self.get_scenes_dirs()

        for i, sequence in enumerate(scenes_dirs):
            seq_frames = []
            seq_annots = []
            for frame in sorted(os.listdir(join(self.dir_data, sequence))):
                # So far we are ignoring the frame dimensions (h, w).
                annot, h, w, valid = get_annotations(self.dir_annot, sequence, frame)
                if valid:
                    seq_frames.append(join(self.dir_data, sequence, frame))
                    seq_annots.append(annot)
                    list_idx.append(i)
            frames.append(seq_frames)
            annotations.append(seq_annots)

        metadata = {'frames': frames,
                    'annotations': annotations,
                    'list_idx': list_idx}

        for key, value in metadata.items():
            setattr(self, key, value)

        return metadata

    def get_pair(self, seq_idx, frame_idx=None):
        """ Gets two frames separated by at most self.max_frame_sep frames (in
        both directions), both contained in the sequence corresponding to the
        given index.

        Args:
            seq_idx: (int) The index of the sequence from which sequence we
                randomly choose one or two frames.
            frame_idx: (int) Optional, the index of the first frame of the
                pair we are constructing

        Returns:
            first_frame_idx: The index of the reference frame inside self.frames
            second_frame_idx: The index of the search frame inside self.frames
        """
        size = len(self.frames[seq_idx])
        if frame_idx is None:
            first_frame_idx = random.randint(0, size-1)
        else:
            first_frame_idx = frame_idx

        min_frame_idx = max(0, (first_frame_idx - self.max_frame_sep))
        max_frame_idx = min(size - 1, (first_frame_idx + self.max_frame_sep))
        second_frame_idx = random.randint(min_frame_idx, max_frame_idx)
        # # Guarantees that the first and second frames are not the same though
        # # it wouldn't be that much of a problem
        # if second_frame_idx == first_frame_idx:
        #     if first_frame_idx == 0:
        #         second_frame_idx += 1
        #     else:
        #         second_frame_idx -= 1
        return first_frame_idx, second_frame_idx

    def ref_context_size(self, h, w):
        """ This function defines the size of the reference region around the
        target as a function of its bounding box dimensions when the context
        margin is added.

        Args:
            h: (int) The height of the bounding box in pixels.
            w: (int) The width of the bounding box in pixels.
        Returns:
            ref_size: (int) The side of the square reference region in pixels.
                Note that it is always an odd integer.
        """
        margin_size = self.cxt_margin*(w + h)
        ref_size = sqrt((w + margin_size) * (h + margin_size))
        # make sur ref_size is an odd number
        ref_size = (ref_size//2)*2 + 1
        return int(ref_size)

    def preprocess_sample(self, seq_idx, first_idx, second_idx):
        """ Loads and preprocesses the inputs and output for the learning problem.
        The input consists of an reference image tensor and a search image tensor,
        the output is the corresponding label tensor. It also outputs the index of
        the sequence from which the pair was chosen as well as the ref and search
        frame indexes in said sequences.

        Args:
            seq_idx: (int) The index of a sequence inside the whole dataset
            first_idx: (int) The index of the reference frame inside
                the sequence.
            second_idx: (int) The index of the search frame inside
                the sequence.

        Returns:
            out_dict: (dict) Dictionary containing all the output information
                stored with the following keys:

                'ref_frame': (torch.Tensor) The reference frame with the
                    specified size.
                'srch_frame': (torch.Tensor) The search frame with the
                    specified size.
                'label': (torch.Tensor) The label created with the specified
                    function in self.label_fcn.
                'seq_idx': (int) The index of the sequence in terms of the self.frames
                    list.
                'ref_idx': (int) The index of the reference image inside the given
                    sequence, also in terms of self.frames.
                'srch_idx': (int) The index of the search image inside the given
                    sequence, also in terms of self.frames.
        """
        reference_frame_path = self.frames[seq_idx][first_idx]
        search_frame_path = self.frames[seq_idx][second_idx]
        ref_annot = self.annotations[seq_idx][first_idx]
        srch_annot = self.annotations[seq_idx][second_idx]

        # Get size of context region for the reference image, as the geometric
        # mean of the dimensions added with a context margin
        ref_w = (ref_annot['xmax'] - ref_annot['xmin'])/2
        ref_h = (ref_annot['ymax'] - ref_annot['ymin'])/2
        ref_ctx_size = self.ref_context_size(ref_h, ref_w)
        ref_cx = (ref_annot['xmax'] + ref_annot['xmin'])/2
        ref_cy = (ref_annot['ymax'] + ref_annot['ymin'])/2

        ref_frame = self.img_read(reference_frame_path)
        ref_frame = np.float32(ref_frame)

        ref_frame, pad_amounts_ref = crop_img(ref_frame, ref_cy, ref_cx, ref_ctx_size)
        try:
            ref_frame = resize_and_pad(ref_frame, self.reference_size, pad_amounts_ref,
                                       reg_s=ref_ctx_size, use_avg=True,
                                       resize_fcn=self.resize_fcn)
        # If any error occurs during the resize_and_pad function we print to
        # the terminal the path to the frame that raised such error.
        except AssertionError:
            print('Fail Ref: ', reference_frame_path)
            raise

        srch_ctx_size = ref_ctx_size * self.search_size / self.reference_size
        srch_ctx_size = (srch_ctx_size//2)*2 + 1

        srch_cx = (srch_annot['xmax'] + srch_annot['xmin'])/2
        srch_cy = (srch_annot['ymax'] + srch_annot['ymin'])/2

        srch_frame = self.img_read(search_frame_path)
        srch_frame = np.float32(srch_frame)
        srch_frame, pad_amounts_srch = crop_img(srch_frame, srch_cy, srch_cx, srch_ctx_size)
        try:
            srch_frame = resize_and_pad(srch_frame, self.search_size, pad_amounts_srch,
                                        reg_s=srch_ctx_size, use_avg=True,
                                        resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Search: ', search_frame_path)
            raise

        if self.label is not None:
            label = self.label
        else:
            label = self.label_fcn(self.final_size, self.pos_thr, self.neg_thr,
                                   upscale_factor=self.upscale_factor)
        # OBS: The ToTensor transform converts the images from a 0-255 range
        # to a 0-1 range
        ref_frame = self.transforms(ref_frame)
        srch_frame = self.transforms(srch_frame)

        out_dict = {'ref_frame': ref_frame, 'srch_frame': srch_frame,
                    'label': label, 'seq_idx': seq_idx, 'ref_idx': first_idx,
                    'srch_idx': second_idx }
        return out_dict

    def __getitem__(self, idx):
        """ Returns the inputs and output for the learning problem. The input
        consists of an reference image tensor and a search image tensor, the
        output is the corresponding label tensor.

        Args:
            idx: (int) The index of a sequence inside the whole dataset, from
                which the function will choose the reference and search frames.

        Returns:
            ref_frame (torch.Tensor): The reference frame with the
                specified size.
            srch_frame (torch.Tensor): The search frame with the
                specified size.
            label (torch.Tensor): The label created with the specified
                function in self.label_fcn.
        """
        seq_idx = self.list_idx[idx]
        first_idx, second_idx = self.get_pair(seq_idx)
        return self.preprocess_sample(seq_idx, first_idx, second_idx)

    def __len__(self):
        return len(self.list_idx)


class ImageNetVID_val(ImageNetVID):
    """ The validation version of the ImageNetVID dataset inherits from the
    train version. The main difference between the two is the way they sample
    the images: while the train version samples in train time, choosing a pair
    of images for the given sequence, the val version chooses all pairs in
    initialization time and stores them internally. The val sampling follows a
    pseudo-random sequence, but since it seeds the random sequence generators,
    the choice is deterministic, so the choice is saved in a file called
    val.metadata that can be charged at initialization time.
    """

    def __init__(self, *args, **kwargs):
        """ For the complete argument description see the Parent class. The val
        version uses the original initialization but it gets a list of pairs of
        indexes consistent with the list of frames and annotations.
        """
        super().__init__(*args, **kwargs)

    def set_root_dirs(self, root):
        self.dir_data = join(root, 'Data', 'VID', "val")
        self.dir_annot = join(root, 'Annotations', 'VID', "val")

    def get_scenes_dirs(self):
        """Apply the right glob function to get the scene directories, relative
        to the data directory
        """
        glob_expression = join(self.dir_data, '*')
        relative_paths = [relpath(p, self.dir_data) for p in sorted(glob.glob(glob_expression))]
        return relative_paths

    def check_metadata(self, metadata):
        """ Checks the val metadata, first by doing the check defined in the
        parent class, but for the val dataset, since the 'frames', 'annotations'
        and 'list_idx' have the same structure for both types.Then it checks
        the specific val elements of the metadata.

        Args:
            metadata: (dict) The metadata dictionary, derived from the json
                metadata file.
        """
        if not super().check_metadata(metadata):
            return False
        if not all(key in metadata for key in ('list_pairs', 'max_frame_sep')):
            return False
        # Check if the maximum frame separation is the same as the metadata one,
        # since the choice of pairs depends on this separation parameter.
        if metadata['max_frame_sep'] != self.max_frame_sep:
            return False
        return True

    def build_metadata(self):
        metadata = super().build_metadata()
        self.list_pairs = self.build_test_pairs()
        metadata['list_pairs'] = self.list_pairs
        metadata['max_frame_sep'] = self.max_frame_sep
        return metadata

    def build_test_pairs(self):
        """ Creates the list of pairs of reference and search images in the
        val dataset. It follows the same general rules for choosing pairs as the
        train dataset, but it does it in initialization time instead of
        train/val time. Moreover it seeds the random module with a fixed seed to
        get the same choice of pairs across different executions.
        Returns:
            list_pairs: (list) A list with all the choosen pairs. It consists
                of a list of tuples with 3 elements each: the index inside the
                sequence of the reference and of the search frame and their
                sequence index. It has one tuple for each frame in the dataset,
                though the function could be easily extended to any number of
                pairs.
        """
        # Seed random lib to get consistent values
        random.seed(100)
        list_pairs = []
        for seq_idx, seq in enumerate(self.frames):
            for frame_idx in range(len(seq)):
                list_pairs.append([seq_idx, *super().get_pair(seq_idx, frame_idx)])
        random.shuffle(list_pairs)
        # Reseed the random module to avoid disrupting any external use of it.
        random.seed()
        return list_pairs

    def __getitem__(self, idx):
        """ Returns the reference and search frame indexes along with the sequence
        index corresponding to the given index 'idx'. The sequence itself is fixed
        from the initialization, so the function simply returns the idx'th element
        of the list_pairs.

        Args:
            idx: (int) The index of the current iteration of validation.
        """
        list_idx, first_idx, second_idx = self.list_pairs[idx]
        return self.preprocess_sample(list_idx, first_idx, second_idx)
