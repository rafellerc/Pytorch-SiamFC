import os
import time
from contextlib import ExitStack
from math import sqrt

import numpy as np

from imageio import imread

from utils.visualization import show_frame_and_response_map
from utils.visualization import save_frame_and_response_map
import matplotlib
# When using a system with no display capabilities we must change the
# Matplotlib backend. Apparently the DISPLAY variable is not set in Windows
# so we must check if we are in a Posix OS
if os.name == 'posix' and "DISPLAY" not in os.environ:
    print('[matplotlib] No display available. Using Agg backend')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
import matplotlib.animation as manimation


# TODO start_frame not used, delete?
def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h,
            final_score_sz, tracking_graph, start_frame, vid_name="Unknown"):
    """ Runs the tracker on the given list of frames and with the given
    parameters.

    Args:
        hp (named_tuple): The tuple containing the hyperparameters.
        run (named_tuple): The tuple containing the run information, such as
            if the user requests visualization or video-making.
        design (named_tuple): The tuple containing the design and architecture
            information, such as the network being used.
        frame_name_list (list): The list containing the full paths to the
            frames being evaluated.
        pos_x, pos_y (int): The initial position of the target in the image.
        target_w, target_h (int): The dimensions of the initial bounding box.
        final_score_sz (int): The desired size of the final score map.
        tracking_graph (TrackingGraph): The tracking graph defining the
            calculations of the tracker.
        vid_name (str): If in make_video mode, the name of the saved video.

    Returns:
        bboxes (list): The list of all the bounding boxes calculated by the
            Tracker, used to calculate its accuracy.
        speed (float): The speed of the tracker in frames per second.
    """

    num_frames = len(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames, 4))

    scale_factors = hp.scale_step**np.linspace(-(hp.scale_num//2),
                                               hp.scale_num//2,
                                               num=hp.scale_num)

    # cosine window to penalize large displacements
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    hann_1d = hann_1d / np.sum(hann_1d)
    penalty = np.transpose(hann_1d) * hann_1d

    # Computation of the context region around the reference, the context region
    # size is equal to the geometric mean of the dimensions of the bounding box
    # added of a context margin (variable 'context').
    # x -> reference, z -> search_region, sz -> size
    context = design.context*(target_w+target_h)
    z_sz = sqrt((target_w+context)*(target_h+context))
    x_sz = float(design.search_sz) / design.reference_sz * z_sz
#
    # thresholds to saturate patches shrinking/growing
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz

    # save first frame position (from ground-truth)
    bboxes[0, :] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h

    crop_z = tracking_graph.preprocess_z(frame_name_list[0], pos_x,
                                         pos_y, z_sz)
    templates_z_ = tracking_graph.get_template_z(crop_z)

    new_templates_z_ = templates_z_

    t_start = time.time()

    # Write video
    if run.make_video:
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title="Test", artist='Rafael Eller',
                        comment='Tracked sequence with score maps')
        writer = FFMpegWriter(fps=5, metadata=metadata, bitrate=1800)
        fig = plt.figure()

    # This with statement has a context manager so it only takes effect if
    # run.make_video is True
    with ExitStack() as stack:
        if run.make_video:
            video_path = run.output_video_folder + vid_name + ".mp4"
            stack.enter_context(writer.saving(fig,video_path, num_frames - 1))
        # Get an image from the queue
        for i, frame in enumerate(frame_name_list[1:]):
            # Since we start the tracking in the second frame we have to shift
            # the index i of one frame.
            idx = i + 1

            scaled_reference = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors

            crops_x = tracking_graph.preprocess_x(frame, pos_x,
                                                  pos_y, z_sz)
            scores = tracking_graph.get_score_map(templates_z_, crops_x)

            # penalize change of scale
            scores[0, :, :] = hp.scale_penalty*scores[0, :, :]
            scores[2, :, :] = hp.scale_penalty*scores[2, :, :]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores, axis=(1, 2)))
            # update scaled sizes
            x_sz = (1-hp.scale_lr) * x_sz + \
                   (hp.scale_lr * scaled_search_area[new_scale_id])
            target_w = (1-hp.scale_lr) * target_w + \
                       (hp.scale_lr * scaled_target_w[new_scale_id])
            target_h = (1-hp.scale_lr) * target_h + \
                       (hp.scale_lr * scaled_target_h[new_scale_id])
            # select response with new_scale_id
            score_ = scores[new_scale_id, :, :]
            score_ = score_ - np.min(score_)
            score_ = score_/np.sum(score_)

            # [Rafael] This is a very weird way of applying the displacement penalty,
            # normally I think of a penalty as an element-wise multiplication
            # of the score by the penalty, not a weighted sum.

            # apply displacement penalty
            score_ = (1-hp.window_influence) * score_ + \
                     (hp.window_influence * penalty)
            pos_x, pos_y = _update_target_position(pos_x, pos_y, score_,
                                                   final_score_sz,
                                                   design.tot_stride,
                                                   design.search_sz,
                                                   hp.response_up, x_sz)
            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            bboxes[idx, :] = (pos_x - target_w/2,
                              pos_y - target_h/2,
                              target_w,
                              target_h)
            # update the target representation with a rolling average
            if hp.z_lr > 0:
                new_crop_z = tracking_graph.preprocess_z(frame,
                                                         pos_x,
                                                         pos_y,
                                                         z_sz)
                new_templates_z_ = tracking_graph.get_template_z(new_crop_z)

                templates_z_ = (1-hp.z_lr) * np.asarray(templates_z_) + \
                    hp.z_lr * np.asarray(new_templates_z_)

            # update template patch size
            z_sz = np.clip((1-hp.scale_lr) * z_sz +
                           hp.scale_lr * scaled_reference[new_scale_id],
                           min_z,
                           max_z)

            if run.visualization:
                image_ = imread(frame)
                norm_score = 255*np.divide(score_ - np.amin(score_),
                                           np.amax(score_))
                show_frame_and_response_map(image_, bboxes[idx, :], 1,
                                            crops_x[new_scale_id],
                                            norm_score, pause=1)
            if run.make_video:
                image_ = imread(frame)
                norm_score = 255*np.divide(score_ - np.amin(score_),
                                           np.amax(score_))
                save_frame_and_response_map(image_, bboxes[idx, :], 1,
                                            crops_x[new_scale_id],
                                            norm_score, writer, fig)

    t_elapsed = time.time() - t_start
    speed = num_frames/t_elapsed

    plt.close('all')

    return bboxes, speed


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride,
                            search_sz, response_up, x_sz):
    """ Updates the position of the target based on the previous position and
    the score map.

    Args:
        pos_x, pos_y (int): The previous position of the target.
        score (np.ndarray): The upscaled score map of the current frame.
        final_score_sz (int): The size of the score map after the upscale.
        tot_stride (int): The total stride of the score map, in relation to the
            original image.
        search_sz (int): The designed size of the search region in pixels.
        response_up (int): The score map upscale factor.
        x_sz (int): the current size of the target context region.

    Returns:
        pos_x, pos_y (int): The new position of the target.
    """
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y
