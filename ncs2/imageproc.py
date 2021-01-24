import logging as log
import os
import re
import shutil
import time
from os import listdir
from os.path import join
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from openvino.inference_engine import IECore
import ngraph as ng


class ImageProc:

    def __init__(self, args):
        if not os.path.exists(args.model):
            log.error(f"{args.model} not found")
            exit(4)
        if not os.path.exists(args.input):
            log.error(f"{args.input} not found")
            exit(4)
        self.args = args
        log.info(f"Creating Inference Engine, device {args.device}")
        self.core = IECore()
        self.images = []
        self.files = []
        self.input = args.input
        self.model = args.model
        self.count = 0

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
                # log.info(f"resize from {image.shape[:-1]} to {(self.args.h, self.args.w)}")
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
        model_path = Path(self.model)
        self.model = model_path.absolute()
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

        # initialize openvino engine
        self.init_engine()
        self.args.network = self.core.load_network(network=self.args.net, device_name=self.args.device)

    def init_engine(self):
        # Plugin initialization for specified device and load extensions library if specified
        # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        log.info(f"Loading network: {self.model}")
        net = self.core.read_network(self.model)
        assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        func = ng.function_from_cnn(net)
        self.args.ops = func.get_ordered_ops()
        #todo: refactor with class
        self.args.input_blob = next(iter(net.input_info))
        self.args.out_blob = next(iter(net.outputs))
        self.args.batch_size, self.args.c, self.args.h, self.args.w = net.input_info[self.args.input_blob].input_data.shape
        log.debug(f"h {self.args.h}, w {self.args.w}")
        self.args.size = (self.args.w, self.args.h)
        self.args.net = net
        return


if __name__ == '__main__':
    img_proc = ImageProc()
    img_proc.prepare()
    img_proc.preprocess_images()

