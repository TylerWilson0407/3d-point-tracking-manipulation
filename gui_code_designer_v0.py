from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from datetime import datetime

import cv2
# import numpy as np

from config import config

import my_gui_test_v0


class ExampleApp(QMainWindow, my_gui_test_v0.Ui_MainWindow):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)

        self.cam_ids = config['cam_ids']

        self.streams = []
        self.stream_threads = []

        self.labels_camfeed = {self.cam_ids[0]: self.label_camfeed1,
                               self.cam_ids[1]: self.label_camfeed2}

        self.labels_fps = {self.cam_ids[0]: self.framerate_cam1,
                           self.cam_ids[1]: self.framerate_cam2}

        self.fps_update_time = datetime.now()

        self.create_streams()

    def create_streams(self):

        for i, cam_id in enumerate(self.cam_ids):
            stream = StreamObject(cam_id)
            thread = QThread()
            self.streams.append(stream)
            self.stream_threads.append(thread)
            stream.moveToThread(thread)
            thread.start()
            stream.signal_cvimage.connect(self.set_pixmap)
            stream.signal_framerate.connect(self.update_fps)
            self.button_start_streams.clicked.connect(stream.start_stream)

    @pyqtSlot(int, object, name='set_pixmap')
    def set_pixmap(self, cam_id, frame):
        # noinspection PyArgumentList
        pixmap = QPixmap.fromImage(convert_to_qimage(frame, swap_rgb=True))
        self.labels_camfeed[cam_id].setPixmap(pixmap)

    @pyqtSlot(int, float, name='update_fps')
    def update_fps(self, cam_id, fps):
        last_update = (datetime.now() - self.fps_update_time).total_seconds()
        if last_update > config['fps_update_interval']:
            self.labels_fps[cam_id].setText('FPS: ' + '%.1f' % fps)
            self.fps_update_time = datetime.now()

    def closeEvent(self, event):
        for stream, thread in zip(self.streams, self.stream_threads):
            stream.end_stream()
            thread.quit()
            while not thread.isFinished():
                pass

        event.accept()


class StreamObject(QObject):

    # signal for emitting the captured frame in OpenCV format(numpy ndarray)
    signal_cvimage = pyqtSignal(int, object)

    # signal for emitting framerate
    signal_framerate = pyqtSignal(int, float)

    def __init__(self, cam_id, parent=None):
        super(self.__class__, self).__init__(parent)
        self.cam_id = cam_id
        self.stream = cv2.VideoCapture(self.cam_id)
        self.time_lastframe = datetime.now()
        self.fps = 0
        self.grabbed = False
        self.stopped = False

    @pyqtSlot(name='start_stream')
    def start_stream(self):
        while not self.stopped:
            self.grabbed, frame = self.stream.read()
            self.update_fps()

            if self.grabbed:
                self.signal_cvimage.emit(self.cam_id, frame)

    def update_fps(self):
        t_delta = datetime.now() - self.time_lastframe
        self.fps = 1 / t_delta.total_seconds()
        self.time_lastframe = datetime.now()
        self.signal_framerate.emit(self.cam_id, self.fps)

    def end_stream(self):
        self.stopped = True


def convert_to_qimage(cv_image, swap_rgb=False):
    """Converts image from opencv(numpy array) into QImage format. Taken
    from StackOverflow answer here: https://stackoverflow.com/a/35857856"""
    height, width, channel = cv_image.shape
    bytes_per_line = width * 3

    if swap_rgb:
        qimage = QImage(cv_image.data, width, height, bytes_per_line,
                        QImage.Format_RGB888).rgbSwapped()
    else:
        qimage = QImage(cv_image.data, width, height, bytes_per_line,
                        QImage.Format_RGB888)

    return qimage


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
