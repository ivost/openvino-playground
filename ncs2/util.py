import logging as log
import os
import re
import shutil
import sys
import time
from os import listdir
from os.path import join
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from openvino.inference_engine import IECore

from ncs2.config import Config


class Util:

    def __init__(self, log_level=log.INFO):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        ap = Config()
        args = ap.parse()
        if not os.path.exists(args.input):
            log.error(f"{args.input} not found")
            exit(0)
        args.t0 = time.perf_counter()
        args.device = "MYRIAD"
        args.verbose = 0
        # accumulates inference time
        args.inference_duration = 0
        args.total = 0
        args.idx = 0
        args.failed = 0
        self.args = args
        log.info("Creating Inference Engine")
        self.core = IECore()
        self.count = 0
        self.images = []
        self.files = []
        self.model = ""

    def count_images(self):
        return self.count_or_load_images(True)

    def load_images(self):
        return self.count_or_load_images(False)

    # todo: change regex to pathlib.glob
    def count_or_load_images(self, count_only):
        path = os.path.abspath(self.args.input)
        pat = None
        if not hasattr(self.args, 're_path') or self.args.re_path is None:
            self.args.re_path = None
        else:
            rex = self.args.re_path
            try:
                pat = re.compile(rex)
            except Exception as err:
                log.error(f"Invalid regex {rex} {err}")
                return 0

        if self.args.re_path is None:
            if not os.path.exists(path):
                return 0
            if os.path.isfile(path):
                if not count_only:
                    if self.args.verbose > 0:
                        log.debug(f"adding image {path}")
                    self.files.append(path)
                return 1
            if not os.path.isdir(path):
                return 0

        count = 0
        limit = self.args.start + self.args.count
        idx = self.args.start
        for f in listdir(path):
            if not count_only and idx >= limit:
                break
            if Path(f).is_dir():
                continue
            if pat is None:
                fp = join(path, f)
                if Path(fp).is_dir():
                    continue
                count += 1
                idx += 1
                if not count_only:
                    if self.args.verbose > 1:
                        log.debug(f"adding image {count}/{limit}  {fp}")
                    self.files.append(fp)
                continue
            # regex
            m = pat.match(f)
            if m is None:
                continue
            fp = join(path, f)
            count += 1
            idx += 1
            if not count_only:
                if self.args.verbose > 1:
                    log.debug(f"adding image {count}/{limit}  {fp}")
                self.files.append(fp)

        if self.args.verbose > 1:
            log.debug(f"{count} images")
        return count

    def preprocess_images(self):
        result = []
        start = time.perf_counter()
        for file in self.files:
            if Path(file).is_dir():
                continue
            if self.args.verbose > 1:
                log.debug(f"file {file}")
            result.append(Image.open(file).convert('RGB').resize(self.args.size, Image.ANTIALIAS))

        duration = (time.perf_counter() - start) / 1000
        if duration > 2:
            log.debug(f"preprocessing took {duration} ms")
        return result

    def preprocess_batch(self, idx):
        self.args.np_images = np.ndarray(shape=(self.args.batch_size, self.args.c, self.args.h, self.args.w))
        for i in range(idx, idx+self.args.batch_size):
            if i >= len(self.files[i]):
                return
            file = self.files[i]
            image = cv2.imread(file)
            if image.shape[:-1] != (self.args.h, self.args.w):
                log.info(f"resize from {image.shape[:-1]} to {(self.args.h, self.args.w)}")
                image = cv2.resize(image, (self.args.w, self.args.h))
            # Change data layout from HWC to CHW
            self.args.np_images[i-idx] = image.transpose((2, 0, 1))
        return

    def copy_to_dir(self, src_file_path, dest_dir_path):
        if not dest_dir_path.exists():
            os.makedirs(dest_dir_path)
        dest = Path(dest_dir_path, src_file_path.name)
        if dest.exists():
            return False
        if self.args.verbose > 1:
            log.debug(f"Copying {src_file_path.name} to {dest_dir_path}")
        shutil.copy2(src_file_path, dest_dir_path)

    def prepare(self):
        # check how many images are available
        cnt = self.count_images()
        if cnt < self.count:
            self.count = cnt
        # Read and pre-process input images
        self.load_images()
        if len(self.files) == 0:
            log.info(f"empty input set")
            exit(0)

        log.info(f"Loaded {len(self.files)} image(s)")
        log.info(f"Image preparation")
        self.model = self.args.model
        assert Path(self.model).exists()
        # initialize openvino engine
        self.init_engine()
        log.info(f"Loading network: {self.model}")
        self.args.network = self.core.load_network(network=self.args.net, device_name=self.args.device)
        log.info(f"device {self.args.device}")
        self.preprocess_images()

    def init_engine(self):
        # Plugin initialization for specified device and load extensions library if specified
        # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        log.debug(f"Loading network: {self.model}")
        self.args.net = self.core.read_network(self.model)
        assert len(self.args.net.input_info.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.args.net.outputs) == 1, "Sample supports only single output topologies"
        self.args.input_blob = next(iter(self.args.net.input_info))
        self.args.out_blob = next(iter(self.args.net.outputs))
        self.args.batch_size, self.args.c, self.args.h, self.args.w = self.args.net.input_info[self.args.input_blob].input_data.shape
        log.debug(f"h {self.args.h}, w {self.args.w}")
        self.args.size = (self.args.w, self.args.h)
        return

    def show_stats(self):
        dur = time.perf_counter() - self.args.t0
        avg = (self.args.inference_duration * 1000) / self.args.total
        log.info(f"  Total images: {self.args.total}, not classified: {self.args.failed}")
        log.info(f"Inference time: {self.args.inference_duration*1000:.0f} ms")
        log.info(f"       Average: {avg:.2f} ms")
        log.info(f"  Elapsed time: {dur*1000:.0f} ms")
        # if out_dir:
        #     log.info(f"Results are in {out_dir}")


if __name__ == '__main__':
    util = Util()
    assert util.args.top == 3

    util.prepare()
    util.preprocess_images()

