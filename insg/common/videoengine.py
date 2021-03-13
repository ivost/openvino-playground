import logging as log
import sys
from time import monotonic

import cv2
import numpy as np
import depthai as dai

from insg.common import Config


class VideoEngine:

    def __init__(self, message, version, config_ini="config.ini", log_level=log.DEBUG):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        log.info(f"\n{message} {version}\n")

        self.pipeline = None
        self.c = Config()
        self.c.read(config_ini)

        n = self.c.network
        self.blob = Config.existing_path(n.blob)
        self.labels = Config.existing_path(n.labels)
        self.input = Config.existing_path(self.c.input.video)
        with open(self.labels, 'r') as file:
            self.labels = [line.split(sep=' ', maxsplit=1)[-1].strip() for line in file]
            log.debug(f"{len(self.labels)} labels")

    def define_pipeline(self):
        # initialize engine
        log.info(f"Initializing depthai pipeline")
        # Start defining a pipeline
        pline = dai.Pipeline()

        # Create neural network input
        xin_nn = pline.createXLinkIn()
        xin_nn.setStreamName("in_nn")

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pline.createNeuralNetwork()
        detection_nn.setBlobPath(str(self.blob))
        xin_nn.out.link(detection_nn.input)

        # Create output
        xout_nn = pline.createXLinkOut()
        xout_nn.setStreamName("nn")
        detection_nn.out.link(xout_nn.input)
        self.pipeline = pline
        return pline

    def run_pipeline(self, pipeline):
        with dai.Device(pipeline) as device:
            # Start pipeline
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            q_in = device.getInputQueue(name="in_nn")
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            # frame = None
            bboxes = []
            labels = []
            confidences = []
            shape = (300, 300)

            # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
            def frame_norm(frame, bbox):
                norm_vals = np.full(len(bbox), frame.shape[0])
                norm_vals[::2] = frame.shape[1]
                return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

            def convert_frame(frame) -> np.ndarray:
                tstamp = monotonic()
                data = cv2.resize(frame, shape).transpose(2, 0, 1).flatten()
                img = dai.ImgFrame()
                img.setData(data)
                img.setTimestamp(tstamp)
                img.setWidth(shape[0])
                img.setHeight(shape[1])
                return img

            cap = cv2.VideoCapture(self.input)
            while cap.isOpened():
                time.sleep(0.005)
                if cv2.waitKey(1) == ord('q'):
                    break
                read_correctly, frame = cap.read()
                if not read_correctly:
                    time.sleep(0.1)
                    continue

                img = convert_frame(dai.ImgFrame())
                q_in.send(img)

                in_nn = q_nn.tryGet()

                if in_nn is None:
                    continue
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

                if frame is None:
                    continue

                # if the frame is available, draw bounding boxes on it and show the frame
                for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                    bbox = frame_norm(frame, raw_bbox)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.putText(frame, self.labels[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                                255)
                    cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                if self.c.output.Preview:
                    cv2.imshow("rgb", frame)


    def model_check(self):
        pass

    def process_classification_results(self, result, idx):
        pass


    def display_result(self, img_file, classes, boxes, ih, iw):
        pass


if __name__ == '__main__':
    # self-test
    engine = VideoEngine("init", "engine", "config.ini")
    pipeline = engine.define_pipeline()
    engine.run_pipeline(pipeline)
