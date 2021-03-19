import datetime
import logging as log
import os.path
import sys
import tempfile
import time
from pathlib import Path

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

        inp = str(self.c.input.video)
        log.debug(f"reading from {inp}")
        cap = cv2.VideoCapture(inp)
        ret, frame = cap.read()
        if not ret:
            log.error("Capture error")
            raise EOFError("Capture error " + inp)

        shape = frame.shape
        size = (shape[1], shape[0])
        log.info(f"frame size {size}")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        n = self.c.network
        self.blob = Config.existing_path(n.blob)
        self.labels = Config.existing_path(n.labels)
        self.confidence = float(self.c.network.confidence)

        # self.input = Config.existing_path(self.c.input.video)
        self.input = self.c.input.video
        # todo: check for stream
        # assert os.path.exists(self.input)
        self.labels = self.create_labels()
        self.exclusions= self.create_exclusions()

        tf = tempfile.NamedTemporaryFile(suffix=".avi")
        self.temp_video = tf.name
        self.output_file = self._output_filename()
        tf.close()
        log.debug(f"Creating VideoWriter: {self.temp_video}, {size}")
        self.video_out = cv2.VideoWriter(self.temp_video, fourcc, 20.0, size)
        return

    def create_exclusions(self):
        self.exclusions = []
        if self.c.network.exclude:
            for n in str(self.c.network.exclude).split(","):
                ok, i = self.safe_label_index(n)
                if ok:
                    self.exclusions.append(i)

        log.debug(f"exclusions: {self.exclusions}")
        return self.exclusions

    def define_pipeline(self):
        # initialize engine
        log.info(f"Initializing depthai pipeline")
        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        # Create neural network input
        xin_nn = self.pipeline.createXLinkIn()
        xin_nn.setStreamName("in_nn")

        # Define a neural network that will make predictions based on the source frames
        detection_nn = self.pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(str(self.blob))
        xin_nn.out.link(detection_nn.input)

        # Create output
        xout_nn = self.pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        detection_nn.out.link(xout_nn.input)
        return

    def run_pipeline(self):
        log.info("Pipeline start")
        with dai.Device(self.pipeline) as device:
            # Start pipeline
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            q_in = device.getInputQueue(name="in_nn")
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
            shape = (300, 300)

            inp = str(self.input)
            log.debug(f"reading from {inp}")
            preview = "true" in str(self.c.output.preview).lower()
            log.debug(f"preview: {preview}")
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
                ok, frame2 = self.process_results(in_nn, frame)
                if preview:
                    if ok:
                        cv2.imshow("rgb", frame2)
                    else:
                        cv2.imshow("rgb", frame)

                # aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
                # frame2 = cv2.resize(self.debug_frame, (int(900), int(900 / aspect_ratio)))
                if self.video_out:
                    if ok:
                        self.video_out.write(frame2)
                    else:
                        # self.video_out.write(frame)
                        # ignore frames w/o results
                        pass

            if self.video_out:
                self.video_out.release()
            assert len(self.output_file) > 0

            log.info(f"Convert to {self.c.output.type}")
            self._convert_to_mp4()
            assert os.path.exists(self.output_file)
            log.info(f"Output file is ready: {self.output_file}")

        log.info("Pipeline end")
        return

    def safe_label_index(self, s: str) -> (bool, int):
        max_index = len(self.labels)
        try:
            n = int(s)
            if max_index > 0 and 0 <= n < max_index:
                return True, n
        except ValueError:
            return False, 0

    def _output_filename(self):
        dir = self.c.output.dir
        assert (os.path.isdir(dir))
        assert (os.path.exists(dir))

        ts = datetime.datetime.now().isoformat(timespec='seconds')
        ts = ts.replace(":", "-")
        if str(self.input).startswith("rtsp:"):
            name = f"rtsp"
        else:
            name = Path(self.input).resolve().name
            assert len(name) > 0
            if '.' in name:
                el = name.split('.')
                name = el[-2]

        fname = f"{name}_{ts}_{self.c.var.name}_{self.c.output.width}x{self.c.output.height}.{self.c.output.type}"

        self.output_file = os.path.join(dir, fname)
        print(f"generate_output_file {self.input} -> {self.output_file}")
        return self.output_file

    def model_check(self):
        pass

    def process_results(self, in_nn, frame):
        # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
        bboxes = np.array(in_nn.getFirstLayerFp16())
        # transform the 1D array into Nx7 matrix
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        # filter out the results which confidence less than a defined threshold
        bboxes = bboxes[bboxes[:, 2] > self.confidence]
        if len(bboxes) == 0:
            return False, None

        # Cut bboxes and labels
        labels = bboxes[:, 1].astype(int)
        confidences = bboxes[:, 2]
        bboxes = bboxes[:, 3:7]
        # log.info(f"process_results conf: {confidence}, {len(bboxes)} bboxes")
        # todo: config
        color_bgr = (0, 250, 250)
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_size = 0.9
        thickness = 4
        count = 0
        for raw_bbox, label, conf in zip(bboxes, labels, confidences):
            if label in self.exclusions:
                continue
            log.debug(f"conf: {conf}, label: {label} - {self.labels[label]}")
            count += 1
            bbox = _frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, thickness)
            cv2.putText(frame, self.labels[label], (bbox[0] + 10, bbox[1] + 20),
                        font, font_size, color_bgr)
            cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                        font, font_size, color_bgr)
        if count:
            return True, frame
        else:
            return False, None

    def _convert_to_mp4(self):
        import subprocess
        inp = self.temp_video
        assert os.path.exists(inp)
        outp = self.output_file
        log.info(f"converting {inp} to {outp}")
        # ffmpeg -i debug.avi -y a.mp4
        res = subprocess.run(["ffmpeg", "-analyzeduration", "1000000", "-i", inp, "-y", outp])
        log.info(str(res))
        if ("returncode=0" in str(res)) and os.path.exists(outp):
            log.debug(f"Deleting {inp}")
            os.remove(inp)
        return

    def create_labels(self):
        with open(self.labels, 'r') as file:
            self.labels = [line.split(sep=' ', maxsplit=1)[-1].strip() for line in file]
            # log.debug(f"{len(self.labels)} labels")
        return self.labels


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
    engine.define_pipeline()
    engine.run_pipeline()
