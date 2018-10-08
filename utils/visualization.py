import os
import numpy as np
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_frame(frame, bbox, fig_n, pause=2):
    plt.ion()
    plt.clf()
    fig = plt.figure(fig_n)
    ax = fig.gca()
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    fig.show()
    fig.canvas.draw()
    plt.pause(pause)


def show_frame_and_response_map(frame, bbox, fig_n, crop_x, score, pause=2):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(131)
    ax.set_title('Tracked sequence')
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Context region')
    ax2.imshow(np.uint8(crop_x))
    ax2.spines['left'].set_position('center')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('center')
    ax2.spines['top'].set_color('none')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(133)
    ax3.set_title('Response map')
    ax3.spines['left'].set_position('center')
    ax3.spines['right'].set_color('none')
    ax3.spines['bottom'].set_position('center')
    ax3.spines['top'].set_color('none')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.imshow(np.uint8(score))

    plt.ion()
    plt.show()
    plt.pause(pause)
    plt.clf()


def save_frame_and_response_map(frame, bbox, fig_n, crop_x, score, writer, fig):
    # fig = plt.figure(fig_n)
    plt.clf()
    ax = fig.add_subplot(131)
    ax.set_title('Tracked sequence')
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Context region')
    ax2.imshow(np.uint8(crop_x))
    ax2.spines['left'].set_position('center')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('center')
    ax2.spines['top'].set_color('none')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(133)
    ax3.set_title('Response map')
    ax3.spines['left'].set_position('center')
    ax3.spines['right'].set_color('none')
    ax3.spines['bottom'].set_position('center')
    ax3.spines['top'].set_color('none')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.imshow(np.uint8(score))

    # ax3.grid()
    writer.grab_frame()


def show_crops(crops, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.uint8(crops[1,:,:,:]))
    ax3.imshow(np.uint8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_scores(scores, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)