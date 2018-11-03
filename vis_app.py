"""
To execute the app first install pyqtgraph, then execute main.py
with the following flags:
   -d: The full path to the dataset folder, the root of the imagenet
folder structure.
   -n: The path to the network to be used, that is, the .pth.tar
file containing the networks weights.
   -t: The type of subset of the dataset, that is, 'train' or 'val'
   -s: The sequence you want to visualize, ranging from 0-3861 for
'train' and 0-554 for 'val'.
   -f: The maximum framerate in fps. The actual framerate will
depend on the processing speed of the computer.

The app is divided in 4 files:
   -main.py: The main script, it parses the user's flags, initializes
the objects, calls the producer thread, which does all the neural
network processing, and execute the display application in the main
thread.
   -display.py: defines the behaviour of the display application,
on what to display the images, how to organize the windows, how to
display the information, etc.
   -producer.py: The app is organized in a threaded producer-consumer
paradigm, so this file defines the behaviour of the producer thread,
which reads the data from the file system and processes it through
the network, then feeding the results to the consumer thread (the
main thread) using a buffer queue.
   -app_utils.py: Defines general functions used in the app.
"""

import argparse
from queue import Queue

from PyQt5 import QtGui, QtCore
import pyqtgraph as pg

from appSiamFC.display import MainUI
from appSiamFC.producer import ProducerThread


pg.setConfigOptions(imageAxisOrder='row-major')

IMAGENET = 'ILSVRC2015'
DEFAULT_MODEL = 'best.pth.tar'


def parse_arguments():
    parser = argparse.ArgumentParser(description="SiamFC app")
    parser.add_argument('-f', '--fps', default=25, dest="fps", type=int,
                        help="The frame per second rate you wish to display the"
                             "video. The true fps rate is displayed FYI.")
    parser.add_argument('-d', '--data_dir', default=IMAGENET,
                        help="Full path to the directory containing the dataset")
    parser.add_argument('-s', '--seq', default=0, dest="seq", type=int,
                        help="The number of the sequence to be displayed."
                             "according to alphabetic order")
    parser.add_argument('-t', '--type', default='train', choices=['train', 'val'],
                        help="The subset of the Imagenet, can be 'train' or"
                        "'val'")
    parser.add_argument('-n', '--net', default=DEFAULT_MODEL,
                        help="Full path the .pth.tar file containing the network's"
                             " weights")
    parser.add_argument('-p', '--prior_width', default=None, type=int,
                        help="The standard deviation of the gaussian displacement"
                        "prior probability that will be applied to the score map,"
                        "centered in the last peak position. The value is given"
                        "in pixels, so the given width corresponds to a contour"
                        "of approximately 0.6 probability. 63 pixels is an overall"
                        "good value, as it represents half of the initial ref"
                        "context_region")
    parser.add_argument('-e', '--exit_on_end', action='store_true', default=False,
                        help="When True this flag exits the program once it has"
                             "finished displaying the frames.")
    parser.add_argument('-b', '--branch', default='alexnet',
                        choices=['alexnet', 'vgg11_5c', 'vgg16_8c'],
                        help="The branch architecture of the siamese net. Should"
                        "correspond to the informed net.")
    parser.add_argument('-c', '--ctx_mode', default='max', choices=['max', 'mean'],
                        help="The strategy used to define the context region around"
                        "the target, using the bounding box dimensions. The 'max'"
                        "mode uses the biggest dimension, while the 'mean' mode"
                        "uses the mean of the dimensions.")
    parser.add_argument('-a', '--alpha', default=0.7, type=restricted_float,
                        help="The alpha value is the proportion of intensity "
                        "of the score map in the mixture with the frame pixels. "
                        "Set alpha to 1 to have only the score map intensity "
                        "displayed, and 0 to have only the sequence's frames "
                        "displayed. Alpha must be between 0 and 1.")
    
    args = parser.parse_args()
    return args


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def update():
    if display.alive:
        display.update()
        app.processEvents()  # force complete redraw for every plot
    else:
        pass


if __name__ == '__main__':
    args = parse_arguments()
    BUFFER = Queue(maxsize=32)
    # Always start by initializing Qt (only once per application)
    app = QtGui.QApplication([])
    # win = QtGui.QMainWindow()
    # win = pg.GraphicsLayoutWidget(show=True, size=(800,800), border=True)
    win = pg.GraphicsLayoutWidget(border=True)

    display = MainUI(win, BUFFER, disp_prior=args.prior_width,
                     exit_on_end=args.exit_on_end, alpha=args.alpha)
    producer = ProducerThread(args.seq, BUFFER, args.data_dir, args.net,
                              set_type=args.type, branch_arch=args.branch,
                              ctx_mode=args.ctx_mode)
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000/args.fps)

    # Start the Producer Thread
    producer.start()
    # Start the application
    QtGui.QApplication.instance().exec_()
