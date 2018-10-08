import numpy as np


def region_to_bbox(region, center=True):
    """
    Transforms the ground-truth annotation to the convenient format. The
    annotations come in different formats depending on the dataset of origin
    (see README, --Dataset--, for details), some use 4 numbers and some use 8
    numbers to describe the bounding boxes.
    """
    n = len(region)
    assert n in [4, 8], ('GT region format is invalid, should have 4 or 8 entries.')

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)


def _rect(region, center):
    """
    Calculate the center if center=True, otherwise the 4 number annotations
    used in the TempleColor and VOT13 datasets are already in the correct
    format.
    (cx, cy) is the center and w, h are the width and height of the target
    When center is False it returns region, which is a 4 tuple containing the
    (x, y) coordinates of the LowerLeft corner and its width and height.
    """
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
        return cx, cy, w, h
    else:
        return region


def _poly(region, center):
    """
    Calculates the center, width and height of the bounding box when the
    annotations are 8 number rotated bounding boxes (used in VOT14 and VOT16).
    Since the Tracker does not try to estimate the rotation of the target, this
    function returns a upright bounding box with the same center width and
    height of the original one.
    The 8 numbers correspond to the (x,y) coordinates of the each of the 4
    corner points of the bounding box.
    """
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1/A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx-w/2, cy-h/2, w, h
