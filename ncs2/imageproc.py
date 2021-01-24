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

from ncs2.config import Config


class ImageProc:

    def __init__(self, args):
        self.args = args
        self.images = []
        self.files = []
        self.input = args.input
        self.count = 0

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
        log.debug(f"Loaded {len(self.files)} image(s)")

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

    def preprocess_images(self, size):
        result = []
        start = time.perf_counter()
        for file in self.files:
            if Path(file).is_dir():
                continue
            if self.args.verbose > 1:
                log.debug(f"file {file}")
            result.append(Image.open(file).convert('RGB').resize(size, Image.ANTIALIAS))

        duration = (time.perf_counter() - start) / 1000
        if duration > 2:
            log.debug(f"preprocessing took {duration} ms")
        return result

    def preprocess_batch(self, idx, batch_size, channels, height, width):
        log.debug(f"preprocess_batch idx {idx}, batch_size {batch_size}")
        np_images = np.ndarray(shape=(batch_size, channels, height, width))
        images_hw = []

        for i in range(idx, idx+batch_size):
            if i >= len(self.files[i]):
                break
            file = self.files[i]
            image = cv2.imread(file)
            ih, iw = image.shape[:-1]
            images_hw.append((ih, iw))
            if image.shape[:-1] != (height, width):
                image = cv2.resize(image, (width, height))
            # Change data layout from HWC to CHW
            np_images[i-idx] = image.transpose((2, 0, 1))
        return np_images, images_hw

    def copy_to_dir(self, src_file_path, dest_dir_path):
        if not dest_dir_path.exists():
            os.makedirs(dest_dir_path)
        dest = Path(dest_dir_path, src_file_path.name)
        if dest.exists():
            return False
        if self.args.verbose > 1:
            log.debug(f"Copying {src_file_path.name} to {dest_dir_path}")
        shutil.copy2(src_file_path, dest_dir_path)


if __name__ == '__main__':
    config = Config()
    args = config.parse()
    img_proc = ImageProc(args)
    img_proc.prepare()
    images = img_proc.preprocess_images((128, 128))
    assert len(images) > 0
    assert len(img_proc.files) >= len(images)

