import re
import shutil
import sys
from pathlib import Path

from PIL import Image
import logging as log
import os
import time
from os import listdir
from os.path import isfile, join
from openvino.inference_engine import IECore
import os
import sys

import cv2
import numpy as np

from arg_parser import ArgParser

class Util:

    def __init__(self):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        ap = ArgParser()
        args = ap.parse_args()
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

    def count_images(self):
        return self.count_or_load_images(True)

    def load_images(self):
        return self.count_or_load_images(False)

    # todo: change regex to pathlib.glob
    def count_or_load_images(self, count_only):
        path = os.path.abspath(self.args.input)
        pat = None
        self.args.files = []
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
                    self.args.files.append(path)
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
                    self.args.files.append(fp)
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
                self.args.files.append(fp)

        if self.args.verbose > 1:
            log.debug(f"{count} images")
        return count

    def preprocess_images(self):
        result = []
        start = time.perf_counter()
        for file in self.args.files:
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
            if i >= len(self.args.files[i]):
                return
            file = self.args.files[i]
            image = cv2.imread(file)
            if image.shape[:-1] != (self.args.h, self.args.w):
                log.debug(f"resize from {image.shape[:-1]} to {(self.args.h, self.args.w)}")
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
        assert Path(self.model).exists()
        log.info(f"Loading network: {self.model}")
        self.args.network = self.core.load_network(network=self.args.net, device_name=self.args.device)
        log.info(f"device {self.device}")
        self.preprocess_images()
        # initialize openvino engine
        self.init_engine()

    def init_engine(self):
        # Plugin initialization for specified device and load extensions library if specified
        # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        log.debug(f"Loading network: {self.model}")
        self.args.net = self.core.read_network(self.model)
        assert len(self.net.input_info.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 1, "Sample supports only single output topologies"
        self.args.input_blob = next(iter(self.args.net.input_info))
        self.args.out_blob = next(iter(self.net.outputs))
        self.args.batch_size, self.args.c, self.args.h, self.w = self.net.input_info[self.args.input_blob].input_data.shape
        log.debug(f"h {self.h}, w {self.w}")
        self.args.size = (self.w, self.h)
        return

    def show_stats(self):
        dur = time.perf_counter() - self.t0
        avg = (self.inference_duration * 1000) / self.total
        log.info(f"  Total images: {self.args.total}, not classified: {self.args.failed}")
        log.info(f"Inference time: {self.inference_duration*1000:.0f} ms")
        log.info(f"       Average: {self:.2f} ms")
        log.info(f"  Elapsed time: {dur*1000:.0f} ms")
        # if out_dir:
        #     log.info(f"Results are in {out_dir}")


def test():
    util = Util()
    assert util.args.top == 3

    util.prepare()
    util.preprocess_images()


if __name__ == '__main__':
    test()


    # def test():
    #    util = Util()
    #    from py.common.arg_parser import parse_args
        # args = parse_args("test")
        # args.verbose = 2
        # args.input = "../../images"
        # args.re_path = R'dog.*\.jpg'
        # args.model = "../../models/squeezenet1.1/FP16/squeezenet1.1.xml"
        # assert Path(args.input).exists()
        # args.count = n = count_images(args)
        # assert 0 < n <= 100
        # args.start = n - 2
        # args.count = 5
        # args.n, args.c, args.h, args.w = 1, 3, 227, 227
        # args.size = (args.w, args.h)
        #
        # load_images(args)
        # m = len(args.files)
        # assert 0 <= m <= 5
        #
        # args.images = preprocess_images(args)
        # for idx in range(len(args.images)):
        #     preprocess_batch(args, idx)
        #     assert args.n == args.np_images.shape[0]
        #
        # log.info("OK")

    # def timeit(method):
    #     def timed(*args, **kw):
    #         ts = time.time()
    #         result = method(*args, **kw)
    #         te = time.time()
    #         if 'log_time' in kw:
    #             name = kw.get('log_name', method.__name__.upper())
    #             kw['log_time'][name] = int((te - ts) * 1000)
    #         else:
    #             print('%r  %6.2f ms' % \
    #                   (method.__name__, (te - ts) * 1000))
    #         return result
