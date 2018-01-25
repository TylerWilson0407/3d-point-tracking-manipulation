import cv2
import numpy as np
from threading import Thread
import time


class CamStream:
    """Utilizes separate thread to stream video from multiple cameras to
    reduce latency."""
    def __init__(self, cam_idx):
        self.cam_idx = cam_idx
        self.frame = None
        self.stopped = False


    def start(self):
        """Create thread and pass self.run to it."""
        self.stream = cv2.VideoCapture(self.cam_idx)
        __, self.frame = self.stream.read()
        Thread(target=self.run, args=(), daemon=True).start()
        return self

    def run(self):
        """Open camera stream and continuously grab frames until stopped."""

        while True:
            if self.stopped:
                return

            __, self.frame = self.stream.read()

    def read(self):

        return self.frame

    def stop(self):

        self.stopped = True
        self.stream.release()


if __name__ == "__main__":
    cams = [0, 1]

    win1 = 'win1'
    win2 = 'win2'
    cv2.namedWindow(win1)
    cv2.namedWindow(win2)

    stream1 = CamStream(0).start()
    stream2 = CamStream(1).start()

    while cv2.waitKey(5) == -1:
        img1 = stream1.read()
        img2 = stream2.read()

        cv2.imshow(win1, img1)
        cv2.imshow(win2, img2)

    cv2.destroyAllWindows()
