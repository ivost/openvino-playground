import cv2
import numpy as np
import logging as log

from ncs2.engine import Engine
from ncs2.stats import Stats


class Detect(Engine):

    def __init__(self):
        super().__init__("Object detection benchmark", "v.2021.1.24", model_override="../models/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.xml")

    def main(self):
        stats = Stats()
        args = self.args
        stats.begin()
        img_proc = self.img_proc
        img_proc.preprocess_images(self.size)
        log.info(f"{len(img_proc.files)} images")
        log.info(f"repeating {args.repeat} time(s)")
        for _ in range(args.repeat):
            print(".", end="", flush=True)
            for idx in range(len(self.img_proc.files)):
                images, images_hw = self.img_proc.preprocess_batch(idx, self.batch_size, self.channels, self.height, self.width)
                data = self.prepare_input(images)
                ###############################
                # inference
                stats.mark()
                res = self.network.infer(inputs=data)
                stats.bump()
                ###############################
        stats.end()
        print("", flush=True)
        log.info(stats.summary())
        self.process_detection_results(res, images_hw)


if __name__ == '__main__':
    d = Detect()
    d.main()
