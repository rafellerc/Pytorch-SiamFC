import xml.etree.ElementTree as ET
import os
from os.path import join, isdir, isfile, splitext
import json
import logging
import shutil

import torch


def get_annotations(annot_dir, sequence_dir, frame_file):
    """ Gets the annotations contained in the xml file of the current frame.

    Args:
        annot_dir (str): The root dir of annotation folders
        sequence_dir(str): The directory in which frame is located, relative to
        root directory
        frame_file (str): The frame filename
        i.e., sequence_identifier/frame_number.JPEG, or the full path.
    Return:
        annotation: (dictionary) A dictionary containing the four values of the
            the bounding box stored in the following keywords:
            'xmax' (int): The rightmost x coordinate of the bounding box.
            'xmin' (int): The leftmost x coordinate of the bounding box.
            'ymax' (int): The uppermost y coordinate of the bounding box.
            'ymin' (int): The lowest y coordinate of the bounding box.

        width (int): The width of the jpeg image in pixels.
        height (int): The height of the jpeg image in pixels.
        valid_frame (bool): False if the target is present in the frame,
            and True otherwise.
        OBS: If the target is not in the frame the bounding box information
        are all None.
    """
    # Separate the sequence from the frame id using the OS path separator
    
    frame_number = splitext(frame_file)[0]
    annot_path = join(annot_dir, sequence_dir, frame_number + '.xml')
    if isfile(annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if root.find('object') is None:
            annotation = {'xmax': None, 'xmin': None, 'ymax': None, 'ymin': None}
            valid_frame = False
            return annotation, width, height, valid_frame
        # TODO Decide what to do with multi-object sequences. For now I'm
        # just going to take the one with track_id = 0
        track_id_zero_present = False
        for obj in root.findall('object'):
            if obj.find('trackid').text == '0':
                bbox = obj.find('bndbox')
                xmax = int(bbox.find('xmax').text)
                xmin = int(bbox.find('xmin').text)
                ymax = int(bbox.find('ymax').text)
                ymin = int(bbox.find('ymin').text)
                track_id_zero_present = True
        if not track_id_zero_present:
            annotation = {'xmax': None, 'xmin': None, 'ymax': None, 'ymin': None}
            valid_frame = False
            return annotation, width, height, valid_frame
    else:
        raise FileNotFoundError("The file {} could not be found"
                                .format(annot_path))

    annotation = {'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin}
    valid_frame = True
    return annotation, width, height, valid_frame


def check_folder_tree(root_dir):
    """ Checks if the folder structure is compatible with the expected one.
    Args:
        root_dir: (str) The path to the root directory of the dataset

    Return:
        bool: True if the structure is compatible, False otherwise.
    """
    data_type = ['Annotations', 'Data']
    dataset_type = ['train', 'val']
    necessary_folders = [join(root_dir, data, 'VID', dataset) for data in data_type for dataset in dataset_type]
    return all(isdir(path) for path in necessary_folders)


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def update_with_dict(self, dictio):
        """ Updates the parameters with the keys and values of a dictionary."""
        self.__dict__.update(dictio)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the
    terminal is saved in a permanent file.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:'
                                                    ' %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept
        # np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such
            as epoch, optimizer state_dict 
        is_best: (bool) True if it is the best
            model seen till now 
        checkpoint: (string) folder where parameters are
            to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}"
              .format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is
    provided, loads state_dict of optimizer assuming it is present in
    checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint) if torch.cuda.is_available() \
        else torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
