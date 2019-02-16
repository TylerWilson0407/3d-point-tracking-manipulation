"""DOCSTRING"""
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

import cv2
import calibration
from config import config as cf
from imutils.video import WebcamVideoStream

import time


# class MainWindow(QMainWindow):
#
#
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)
#
#         self.counter = 0
#
#         layout = QVBoxLayout()
#
#         self.l = QLabel("Start")
#         b = QPushButton("DANGER!")
#         b.pressed.connect(self.oh_no)
#
#         layout.addWidget(self.l)
#         layout.addWidget(b)
#
#         w = QWidget()
#         w.setLayout(layout)
#
#         self.setCentralWidget(w)
#
#         self.show()
#
#         self.timer = QTimer()
#         self.timer.setInterval(1000)
#         self.timer.timeout.connect(self.recurring_timer)
#         self.timer.start()
#
#     def oh_no(self):
#         time.sleep(5)
#
#     def recurring_timer(self):
#         self.counter +=1
#         self.l.setText("Counter: %d" % self.counter)


class StreamObject(QObject):

    # define signal.
    change_pixmap = pyqtSignal(QImage)

    def __init__(self, cam_id):
        super().__init__()
        self.cam_id = cam_id
        self.stream = cv2.VideoCapture(self.cam_id)
        self.ret = None
        self.frame = None

    @pyqtSlot(QImage)
    def update(self):
        while True:
            self.ret, self.frame = self.stream.read()
            if self.ret:
                self.change_pixmap.emit(convert_to_qimage(self.frame))


def convert_to_qimage(cv_image):
    """Converts image from opencv(numpy array) into QImage format. Taken
    from StackOverflow answer here: https://stackoverflow.com/a/35857856"""
    height, width, channel = cv_image.shape
    bytes_per_line = width * 3
    qimage = QImage(cv_image.data, width, height, bytes_per_line,
                    QImage.Format_RGB888)
    return qimage


class QStream(QThread):

    def __init__(self, cam_id):
        super().__init__(self)
        self.cam_id = cam_id
        self.image_signal = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        while True:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convert_to_qt_format = QImage(rgb_image.data,
                                              rgb_image.shape[1],
                                              rgb_image.shape[0],
                                              QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.image_signal.emit(p)
                print('test')


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        print('test')
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        # th = StreamObject(0)
        # th.change_pixmap.connect(self.setImage)

        self.obj = StreamObject(0)
        self.thread = QThread()
        self.obj.moveToThread(self.thread)
        self.thread.started.connect(self.obj.update())
        self.thread.start()

        self.show()


def using_move_to_thread():
    app = QCoreApplication([])
    objThread = QThread()
    obj = StreamObject(0)
    obj.moveToThread(objThread)
    # obj.finished.connect(objThread.quit)
    # objThread.started.connect(obj.update())
    # objThread.finished.connect(app.exit)
    objThread.start()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # ex = App()
    # sys.exit(app.exec_())

    # using_move_to_thread()

    app = QApplication(sys.argv)
    widget = App()
    # obj = StreamObject(0)
    # thread = QThread()
    # obj.moveToThread(thread)
    # thread.started.connect(obj.update())
    # thread.start()
    print('test')
    sys.exit(app.exec_())
