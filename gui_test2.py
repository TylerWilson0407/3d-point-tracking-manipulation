from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

import cv2
import numpy as np

from config import config

import my_gui_test_v0


class Example(QObject):
    signalStatus = pyqtSignal(str)

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.cam_ids = config['cam_ids']

        # Create a gui object.
        self.gui = Window()

        # Create a new worker thread.
        self.create_stream_threads()

        # Make any cross object connections.
        self.connect_signals()

        self.gui.show()

    def connect_signals(self):
        self.gui.button_cancel.clicked.connect(self.force_stream_reset)
        # self.signalStatus.connect(self.gui.updateStatus)
        self.parent().aboutToQuit.connect(self.force_stream_quit)

    def create_stream_threads(self):

        # Setup the worker object and the worker_thread.
        self.stream_object = StreamObject(0)
        self.stream_thread = QThread()
        self.stream_object.moveToThread(self.stream_thread)
        self.stream_thread.start()

        # Connect any worker signals
        # self.stream_object.signalStatus.connect(self.gui.updateStatus)

        self.stream_object.signalImage.connect(self.gui.updateImage)
        self.stream_object.signalArray.connect(self.gui.updateArray)

        self.gui.button_start.clicked.connect(self.stream_object.start_stream)

    def force_stream_reset(self):
        if self.stream_thread.isRunning():
            print('Terminating thread.')
            self.stream_thread.terminate()

            print('Waiting for thread termination.')
            self.stream_thread.wait()

            self.signalStatus.emit('Idle.')

            print('building new working object.')
            self.create_stream_threads()

    def force_stream_quit(self):
        if self.stream_thread.isRunning():
            self.stream_thread.terminate()
            self.stream_thread.wait()


class ExampleApp(QMainWindow, my_gui_test_v0.Ui_MainWindow):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)

        self.cam_ids = config['cam_ids']

        # Create a gui object.
        self.gui = Window()

        # Create a new worker thread.
        self.create_stream_threads()

        # Make any cross object connections.
        self.connect_signals()

        self.gui.show()

    def connect_signals(self):
        self.gui.button_cancel.clicked.connect(self.force_stream_reset)
        # self.signalStatus.connect(self.gui.updateStatus)
        self.parent().aboutToQuit.connect(self.force_stream_quit)

    def create_stream_threads(self):

        # Setup the worker object and the worker_thread.
        self.stream_object = StreamObject(0)
        self.stream_thread = QThread()
        self.stream_object.moveToThread(self.stream_thread)
        self.stream_thread.start()

        # Connect any worker signals
        # self.stream_object.signalStatus.connect(self.gui.updateStatus)

        self.stream_object.signalImage.connect(self.gui.updateImage)
        self.stream_object.signalArray.connect(self.gui.updateArray)

        self.gui.button_start.clicked.connect(self.stream_object.start_stream)

    def force_stream_reset(self):
        if self.stream_thread.isRunning():
            print('Terminating thread.')
            self.stream_thread.terminate()

            print('Waiting for thread termination.')
            self.stream_thread.wait()

            self.signalStatus.emit('Idle.')

            print('building new working object.')
            self.create_stream_threads()

    def force_stream_quit(self):
        if self.stream_thread.isRunning():
            self.stream_thread.terminate()
            self.stream_thread.wait()

    @pyqtSlot(QImage)
    def updateImage(self, image):
        self.image_labels[0].setPixmap(QPixmap.fromImage(image))
        self.image_labels[1].setPixmap(QPixmap.fromImage(image))


class StreamObject(QObject):

    signalStatus = pyqtSignal(str)

    signalImage = pyqtSignal(QImage)
    signalArray = pyqtSignal(np.ndarray)

    def __init__(self, cam_id, parent=None):
        super(self.__class__, self).__init__(parent)
        self.cam = cv2.VideoCapture(cam_id)

    @pyqtSlot()
    def start_stream(self):
        while True:
            _, frame = self.cam.read()
            self.signalImage.emit(convert_to_qimage(frame, swap_rgb=True))
            self.signalArray.emit(frame)


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


class Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.button_start = QPushButton('Start', self)
        self.button_cancel = QPushButton('Cancel', self)
        # self.label_status = QLabel('', self)

        self.numpy_label = QLabel(self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.button_start)
        layout.addWidget(self.button_cancel)
        # layout.addWidget(self.label_status)

        layout.addWidget(self.numpy_label)

        self.cam_ids = config['cam_ids']
        self.image_labels = []

        for _ in self.cam_ids:
            img_label = QLabel(self)
            self.image_labels.append(img_label)
            img_label.resize(640, 480)
            layout.addWidget(img_label)


    # @pyqtSlot(str)
    # def updateStatus(self, status):
    #     self.label_status.setText(status)

    @pyqtSlot(QImage)
    def updateImage(self, image):
        self.image_labels[0].setPixmap(QPixmap.fromImage(image))
        self.image_labels[1].setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(object)
    def updateArray(self, frame):
        self.numpy_label.setText(str(frame[0][0][0]))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = Example(app)
    sys.exit(app.exec_())
