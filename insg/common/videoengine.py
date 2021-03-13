import logging as log
import sys
import time

import cv2
import depthai as dai
import numpy as np

from insg.common import Config


class VideoEngine:

    def __init__(self, message, version, config_ini="config.ini", log_level=log.DEBUG):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        log.info(f"\n{message} {version}\n")

        self.pipeline = None
        self.c = Config()
        self.c.read(config_ini)
        # todo: config
        width = 1920
        height = 1080
        size = (width, height)
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # self.video_out = cv2.VideoWriter('debug.mp4', fourcc, 20.0, size)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_out = cv2.VideoWriter('debug.avi', fourcc, 20.0, size)

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
            shape = (300, 300)

            inp = str(self.input)
            log.debug(f"reading from {inp}")
            cap = cv2.VideoCapture(inp)
            while cap.isOpened():
                if cv2.waitKey(1) == ord('q'):
                    break
                read_correctly, frame = cap.read()
                if not read_correctly or frame is None:
                    break

                img = dai.ImgFrame()
                img = _convert_frame(frame, img, shape)
                q_in.send(img)

                in_nn = q_nn.tryGet()
                if in_nn is None:
                    time.sleep(0.005)
                    continue
                frame = self.process_results(in_nn, frame)

                if self.c.output.preview:
                    cv2.imshow("rgb", frame)

                # aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
                # frame2 = cv2.resize(self.debug_frame, (int(900), int(900 / aspect_ratio)))
                if self.video_out:
                    # print(f"=== writing frame, aspect ratio {aspect_ratio} ")
                    self.video_out.write(frame)

            if self.video_out:
                self.video_out.release()

    def model_check(self):
        pass

    def process_results(self, in_nn, frame):
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
        color_bgr = (0, 250, 250)
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_size = 0.5
        thickness = 2
        for raw_bbox, label, conf in zip(bboxes, labels, confidences):
            bbox = _frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, thickness)
            cv2.putText(frame, self.labels[label], (bbox[0] + 10, bbox[1] + 20),
                        font, font_size, color_bgr)
            cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                        font, font_size, color_bgr)
        return frame


def _convert_frame(frame, img, shape) -> np.ndarray:
    tstamp = time.monotonic()
    data = cv2.resize(frame, shape).transpose(2, 0, 1).flatten()
    img.setData(data)
    img.setTimestamp(tstamp)
    img.setWidth(shape[0])
    img.setHeight(shape[1])
    return img


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def _frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


if __name__ == '__main__':
    # self-test
    engine = VideoEngine("init", "engine", "config.ini")
    pipeline = engine.define_pipeline()
    engine.run_pipeline(pipeline)
