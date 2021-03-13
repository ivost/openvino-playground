import logging as log
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
from time import monotonic
import tkinter as tk
from tkinter import filedialog

import sys
import importlib.util
import pkgutil
from pkgutil import extend_path
import importlib.util

import insg
import insg.common
import insg.oak

# from .. common.videoengine import VideoEngine
# import common.videoengine

cwd = Path().resolve()
print("cwd", cwd)

# root_dir = cwd.parent.parent.parent.resolve().absolute()
root_dir = "/Users/ivo/github/myriad-playground/insg"
print("root_dir", root_dir)
assert Path(root_dir).exists()


# sys.path.append(root_dir)
# sys.path.append(root_dir + "/common")
# sys.path.append(root_dir + "/oak")

# name = 'insg'
#
# if name in sys.modules:
#     print(f"{name!r} already in sys.modules")
# else:
#     spec = importlib.util.spec_from_file_location(name, root_dir)
#     eng = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(eng)

# elif (spec := importlib.util.find_spec(name)) is not None:
#     # If you chose to perform the actual import ...
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[name] = module
#     spec.loader.exec_module(module)
#     print(f"{name!r} has been imported")
# else:
#     print(f"can't find the {name!r} module")

# __path__ = extend_path('/Users/ivo/github/myriad-playground/insg', 'insg')
# print("__path__", __path__)

# sys.path.append(root_dir)
# sys.path.append(root_dir + "/common")
# sys.path.append(root_dir + "/oak")
# print("sys.path", sys.path)
# wkd = Path(__file__)
# mobilenet_path = str((wkd / Path('../../models/ssdlite_mobilenet_v2/mobilenet.blob')).resolve().absolute())

from insg.common.videoengine import VideoEngine


class Detect(VideoEngine):

    def __init__(self, log_level=log.DEBUG):
        super().__init__("Object detection benchmark", "v.2021.1.25", "detect.ini", log_level=log_level)

    def main(self):
        root_dir = Path(__file__).parent.parent.parent.resolve().absolute()
        mobilenet_path = str((root_dir.joinpath('models/ssdlite_mobilenet_v2/mobilenet.blob')))
        video_path = str((root_dir.joinpath("videos/airport-01-HD.mp4")))

        if len(sys.argv) == 1:
            pass
            # root = tk.Tk()
            # root.withdraw()
            # video_path = filedialog.askopenfilename(initialdir=video_path)
        else:
            if len(sys.argv) > 1:
                video_path = sys.argv[1]
            if len(sys.argv) > 2:
                mobilenet_path = sys.argv[2]

        # Start defining a pipeline
        pipeline = dai.Pipeline()

        # Create neural network input
        xin_nn = pipeline.createXLinkIn()
        xin_nn.setStreamName("in_nn")

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(mobilenet_path)
        xin_nn.out.link(detection_nn.input)

        # Create output
        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        detection_nn.out.link(xout_nn.input)

        # MobilenetSSD label texts
        texts = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                 "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                 "tvmonitor"]

        # Pipeline defined, now the device is connected to
        with dai.Device(pipeline) as device:
            # Start pipeline
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            q_in = device.getInputQueue(name="in_nn")
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            bboxes = []
            labels = []
            confidences = []

            # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
            def frame_norm(frame, bbox):
                norm_vals = np.full(len(bbox), frame.shape[0])
                norm_vals[::2] = frame.shape[1]
                return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

            def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
                return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                read_correctly, frame = cap.read()
                if not read_correctly:
                    break

                tstamp = monotonic()
                data = to_planar(frame, (300, 300))
                img = dai.ImgFrame()
                img.setData(data)
                img.setTimestamp(tstamp)
                img.setWidth(300)
                img.setHeight(300)
                q_in.send(img)

                in_nn = q_nn.tryGet()

                if in_nn is not None:
                    # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
                    bboxes = np.array(in_nn.getFirstLayerFp16())
                    # transform the 1D array into Nx7 matrix
                    bboxes = bboxes.reshape((bboxes.size // 7, 7))
                    # filter out the results which confidence less than a defined threshold
                    bboxes = bboxes[bboxes[:, 2] > 0.5]
                    # Cut bboxes and labels
                    labels = bboxes[:, 1].astype(int)
                    confidences = bboxes[:, 2]
                    bboxes = bboxes[:, 3:7]

                if frame is not None:
                    # if the frame is available, draw bounding boxes on it and show the frame
                    for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                        bbox = frame_norm(frame, raw_bbox)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        cv2.putText(frame, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                                    255)
                        cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.imshow("rgb", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

print(__name__)

if __name__ == '__main__':
    # print(__file__)
    root_dir = Path(__file__).parent.parent.parent.resolve().absolute()
    # sys.path.append(root_dir / "insg")
    # sys.path.append(root_dir / "insg" / "common")
    # sys.path.append(root_dir / "insg" / "oak")
    print(sys.path)
    # print(root_dir)
    path = root_dir / "models" / "ssdlite_mobilenet_v2" / "mobilenet.blob"
    assert path.exists()
    # c = Detect()
    # c.main()

'''
'/Users/ivo/opt/anaconda3/lib/python38.zip', '/Users/ivo/opt/anaconda3/lib/python3.8', 
'/Users/ivo/opt/anaconda3/lib/python3.8/lib-dynload', '/Users/ivo/opt/anaconda3/lib/python3.8/site-packages', 
'/Users/ivo/opt/anaconda3/lib/python3.8/site-packages/aeosa']
'''