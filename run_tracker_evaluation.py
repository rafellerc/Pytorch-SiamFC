#!/usr/bin/env python3
import sys
import os
import argparse
import glob

import numpy as np
from PIL import Image

from tracking.tracker import tracker
from tracking.get_parameters import get_parameters
from tracking.tracking_graph import TrackingGraph
from utils.bbox_transforms import region_to_bbox

# TODO implement a logging to save all the results inside the experiment's folder

# TODO The operations implemented in the Pytorch Version are slightly different
# from the ones in the TensorFlow version, so we must make a new hyperparameter
# search.

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='default', help=("Name of the directory"
                    "containing the tracker's parameters, contained in"
                    "root/tracking/experiments"))

def main():
    args = parser.parse_args()
    hp, evaluation, run, env, design = get_parameters()

    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    tracking_graph = TrackingGraph(design, env, hp)

    # iterate through all videos of evaluation.dataset
    if evaluation.video == 'all':
        dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
        videos_list = list(os.listdir(dataset_folder))
        videos_list = sorted(videos_list)
        nv = len(videos_list)
        speed = np.zeros([nv, evaluation.n_subseq])
        precisions = np.zeros([nv, evaluation.n_subseq])
        precisions_auc = np.zeros([nv, evaluation.n_subseq])
        ious = np.zeros([nv, evaluation.n_subseq])
        lengths = np.zeros([nv, evaluation.n_subseq])
        for i, video in enumerate(videos_list):
            gt, frame_name_list, frame_sz, n_frames = _init_video(env,
                                                                  evaluation,
                                                                  video)
            starts = np.uint8(np.round(np.linspace(0, n_frames - 1,
                                          evaluation.n_subseq,
                                          endpoint=False)))
            for j, start_frame in enumerate(starts):
                current_gt = gt[start_frame:]
                frame_name_list_ = frame_name_list[start_frame:]
                pos_x, pos_y, target_w, target_h = region_to_bbox(current_gt[0])
                bboxes, speed[i,j] = tracker(hp, run, design, frame_name_list_,
                                             pos_x, pos_y, target_w, target_h,
                                             final_score_sz, tracking_graph,
                                             start_frame,
                                             vid_name=video)
                lengths[i,j], precisions[i,j], precisions_auc[i,j], ious[i,j] \
                    = _compile_results(current_gt, bboxes, evaluation.dist_threshold)
                print('{} -- {}'
                      ' -- Precision: {:.2f}'
                      ' -- Precision AUC: {:.2f}'
                      ' -- IOU: {:.2f}'
                      ' -- Speed: {:.2f} --'.format(
                            i, video, precisions[i,j], precisions_auc[i,j],
                            ious[i,j], speed[i,j]))

        # TODO save all the results to a json inside the experiment's folder

        tot_frames = np.sum(lengths)
        mean_precision = np.sum(precisions * lengths) / tot_frames
        mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
        mean_iou = np.sum(ious * lengths) / tot_frames
        mean_speed = np.sum(speed * lengths) / tot_frames
        print(' -- Overall stats (averaged per frame) on {} videos'
              '({} frames) --\n'
              ' -- Precision ({} px): {}'
              ' -- Precision AUC: {:.2f}'
              ' -- IOU: {:.2f}'
              ' -- Speed: {:.2f} --'.format(
                  nv, tot_frames, evaluation.dist_threshold, mean_precision,
                  mean_precision_auc, mean_iou, mean_speed))

    else:
        gt, frame_name_list, _, _ = _init_video(env, evaluation,

                                                evaluation.video)
        pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation
                                                          .start_frame])
        bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y,
                                target_w, target_h, final_score_sz,
                                tracking_graph, evaluation.start_frame)
        _, precision, precision_auc, iou = _compile_results(gt, bboxes,
                                                            evaluation
                                                            .dist_threshold)
        print('{} -- Precision ({} px): {}'
              ' -- Precision AUC: {:.2f}'
              ' -- IOU: {:.2f} '
              ' -- Speed: {:.2f}'.format(
                  evaluation.dist_threshold, precision, precision_auc,
                  iou, speed))


def _compile_results(gt, bboxes, dist_threshold):
    """ Computes the metrics for the evaluation of the tracker in a sequence of
    the dataset.
    Args:
        gt: (numpy.ndarray) The array containing the ground-truths of the given
            video sequence.
        bboxes: (numpy.ndarray) The array containing the bounding boxes for
            each frame predicted by the tracker.
        dist_threshold: (int) A distance threshold in pixels that defines what
            is considered a success and what is considered a failure in terms
            of center location error (distance between the predicted center and
            the ground-truth center).

    Returns:
        lenght: (int) The lenght of the sequence.
        precision: (float) The percentage of frames in the sequence with a
            center location error of less than the given distance threshold.
        precision_auc: (float) The area under the curve of precision measured
            for different values of distance threshold.
        iou: (float) The intersection over union between the predicted bounding
            box and the ground-truth (in percentage terms).
    """
    length = bboxes.shape[0]
    gt4 = np.zeros((length, 4))
    new_distances = np.zeros(length)
    new_ious = np.zeros(length)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(length):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior
    # to given threshold? (OTB metric)
    precision = (new_distances < dist_threshold).mean()*100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[1:]
    thresholds = thresholds.reshape(n_thresholds, 1)
    precisions_ths = (new_distances < thresholds).mean(axis=1)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return length, precision, precision_auc, iou


def _init_video(env, evaluation, video):
    """ Collects all the needed information about the a given video, that is,
    The annotations, the list of frames, and frame size.

    Args:
        env: (namedtuple) The environment parameters contained in the
            environment.json file.
        evaluation: (namedtuple) The evaluation parameters contained in the
            evaluation.json file.
        video: (string) The name of the video folder.

    Returns:
        gt: (numpy.ndarray) The array containing the ground-truth annotations
            for the current video.
        frame_name_list: (list) A list with the basename of each of the frames
            in the video folder.
        frame_sz: (numpy.ndarray) The width and height of the frames of the
            current video.
        n_frames: (int) Total number of frames in the video.
    """
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = glob.glob(os.path.join(video_folder, "*.jpg"))
    frame_name_list = sorted(frame_name_list)
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


if __name__ == '__main__':
    main()
