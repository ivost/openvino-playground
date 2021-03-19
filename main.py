import sys
import time
import cv2
import numpy as np

from insg.common import Stats


def testStats():
    # print(sys.path)
    s = Stats()
    s.begin()
    s.mark()
    time.sleep(0.03)
    s.bump()
    s.mark()
    time.sleep(0.04)
    s.bump(is_error=True)
    s.end()
    print(s)


def test_rtsp():
    cap = cv2.VideoCapture("rtsp://192.168.1.129:554/channel1")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # size = (2304, 1280)
    ret, frame = cap.read()
    shape = frame.shape
    # size = (1920, 1056)
    size = (shape[1], shape[0])
    print("frame size", size)
    video_out = cv2.VideoWriter("./rtsp.mkv", fourcc, 20.0, size)
    while True:
        if cv2.waitKey(1) == ord('q'):
            break
        ret, frame = cap.read()
        if not ret:
            print("error")
            continue
        cv2.imshow('VIDEO', frame)
        video_out.write(frame)

    video_out.release()


if __name__ == "__main__":
    # testStats()
    test_rtsp()
