import sys

import numpy as np
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from matplotlib import cm

from appSiamFC.app_utils import make_gaussian_map, rgb2gray


class MainUI(object):

    def __init__(self, MainWindow, buffer, disp_prior=None, exit_on_end=False,
                 alpha=0.7):
        """
        """
        self.alive = True
        self.disp_prior = disp_prior
        self.exit_on_end = exit_on_end

        MainWindow.setWindowTitle('SiamFC - Demo')
        MainWindow.resize(1000, 900)

        # Define the ViewBoxes for each of the images to be displayed
        self.score_box = MainWindow.addViewBox(1, 0, colspan=3)
        self.gt_box = MainWindow.addViewBox(3, 0, colspan=2)
        self.ref_box = MainWindow.addViewBox(3, 2)
        self.score_box.invertY(True)  # Images usually have their Y-axis pointing downward
        self.gt_box.invertY(True)
        self.ref_box.invertY(True)
        self.score_box.setAspectLocked(True)
        self.gt_box.setAspectLocked(True)
        self.ref_box.setAspectLocked(True)

        self.fpsLabel = pg.LabelItem(justify='left')
        MainWindow.addItem(self.fpsLabel, 0, 0)
        self.visibleLabel = MainWindow.addLabel('', 0, 1)
        self.bufferLabel = MainWindow.addLabel('', 0, 2)
        self.nameLabel = MainWindow.addLabel('', 4, 0, colspan=3)
        font = QtGui.QFont()
        font.setPointSize(4)
        self.nameLabel.setFont(font)

        self.score_img = pg.ImageItem()
        self.gt_img = pg.ImageItem()
        self.ref_img = pg.ImageItem()
        self.score_box.addItem(self.score_img)
        self.gt_box.addItem(self.gt_img)
        self.ref_box.addItem(self.ref_img)
        # self.view_box.setRange(QtCore.QRectF(0, 0, 512, 512))
        self.bounding_box = QtWidgets.QGraphicsRectItem()
        self.bounding_box.setPen(QtGui.QColor(255, 0, 0))
        self.bounding_box.setParentItem(self.gt_img)
        self.gt_box.addItem(self.bounding_box)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        self.peak = pg.GraphItem(size=30, symbol='+', pxMode=True,
                                 symbolBrush=brush,
                                 symbolPen=None)
        self.peak.setParentItem(self.score_img)
        self.score_box.addItem(self.peak)
        self.peak_pos = None
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, alpha=0))
        self.prior_radius = pg.GraphItem(size=0, symbol='o', pxMode=True,
                                         symbolBrush=brush, symbolPen='b')
        self.prior_radius.setParentItem(self.score_img)
        self.score_box.addItem(self.prior_radius)

        # Add the Labels to the images
        param_dict = {'color':(255,255,255),
                      'anchor':(0,1)}
        label_score = pg.TextItem(text='Score Map', **param_dict)
        label_gt = pg.TextItem(text='Ground Truth', **param_dict)
        label_ref = pg.TextItem(text='Reference Image', **param_dict)
        font.setPointSize(16)
        label_score.setFont(font)
        label_gt.setFont(font)
        label_ref.setFont(font)
        label_score.setParentItem(self.score_img)
        label_gt.setParentItem(self.gt_img)
        label_ref.setParentItem(self.ref_img)
        self.score_box.addItem(label_score)
        self.gt_box.addItem(label_gt)
        self.ref_box.addItem(label_ref)

        # The alpha parameter is used to overlay the score map with the image,
        # where alpha=1 corresponds to the score_map alone and alpha=0 is
        # the image alone.
        self.alpha = alpha

        self.error_plot = MainWindow.addPlot(5, 0, colspan=3,
                                             title='Center Error (pixels)')
        self.curve = self.error_plot.plot(pen='y')
        # Sets a line indicating the 63 pixel error corresponding to half of the
        # initial reference bounding box, and a possible measure of tracking
        # failure.
        half_ref = pg.InfiniteLine(movable=False, angle=0, pen=(0, 0, 200),
                                   label='ctr_error={value:0.2f}px',
                                   labelOpts={'color': (200,200,200),
                                              'movable': True,
                                              'fill': (0, 0, 200, 100)})
        half_ref.setPos([63, 63])
        self.error_plot.addItem(half_ref)
        self.center_errors = []

        self.index = 0

        MainWindow.show()

        self.lastTime = ptime.time()
        self.fps = None
        self.buffer = buffer

    def exit(self):
        sys.exit()

    def update(self):
        """
        """
        # Gets an element from the buffer and free the buffer
        buffer_element = self.buffer.get()
        self.buffer.task_done()
        # When the Producer Thread finishes publishing the data it sends a None
        # through the buffer to sinalize it has finished.
        if buffer_element is not None:
            score_map = buffer_element.score_map
            gt_img = buffer_element.img
            ref_img = buffer_element.ref_img
            visible = buffer_element.visible
            name = buffer_element.name
            bbox = buffer_element.bbox

            if self.peak_pos is not None and self.disp_prior is not None:
                h, w = score_map.shape
                disp_prior = make_gaussian_map((h,w), self.peak_pos, self.disp_prior)
                np.expand_dims(disp_prior, axis=2)
                score_map = score_map*disp_prior
                self.prior_radius.setData(pos=[(self.peak_pos[1], self.peak_pos[0])], size=self.disp_prior)

            # Find peak of the score map. You must incorporate all priors before
            # taking the max.
            peak = np.unravel_index(score_map.argmax(), score_map.shape)
            self.peak_pos = peak
            # Apply the inferno color map to the score map.
            # The output of cm.inferno has 4 channels, 3 color channels and a
            # transparency channel
            score_img = cm.inferno(score_map)[:, :, 0:3]
            # Overlay the score_img with a grayscale version of the original
            # frame
            img_gray = rgb2gray(gt_img)
            score_img = score_img[0:img_gray.shape[0], 0:img_gray.shape[1], :]
            score_img = score_img*self.alpha + (1-self.alpha)*img_gray/255

            vis_color = 'g' if visible else 'r'
            self.score_img.setImage(score_img, autoDownsample=False)
            # Set the marker in the peak. The pyqtgraph GraphItem takes the
            # position in terms of the x and y coordinates.
            self.peak.setData(pos=[(peak[1], peak[0])])
            self.gt_img.setImage(gt_img, autoDownsample=False)
            if bbox is not None:
                self.bounding_box.setRect(*bbox)
                center_error = np.linalg.norm([bbox[0]+bbox[2]/2-peak[1], bbox[1]+bbox[3]/2-peak[0]])
                self.center_errors.append(center_error)
                self.curve.setData(self.center_errors)
            else:
                self.bounding_box.setRect(0, 0, 0, 0)
            self.ref_img.setImage(ref_img, autoDownsample=False)
            # Calculate the fps rate.
            now = ptime.time()
            dt = now - self.lastTime
            self.lastTime = now
            if self.fps is None:
                self.fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                self.fps = self.fps * (1-s) + (1.0/dt) * s
            self.fpsLabel.setText('{:.2f} fps'.format(self.fps), color='w')
            self.visibleLabel.setText('Visible: {}'.format(visible), color=vis_color)
            self.bufferLabel.setText('{} in Buffer'.format(self.buffer.qsize()))
            self.nameLabel.setText(name, size='10pt', color='w')

            self.index += 1
        else:
            # Set alive attribute to False to indicate the end of the program
            self.alive = False
            if self.exit_on_end:
                self.exit()
