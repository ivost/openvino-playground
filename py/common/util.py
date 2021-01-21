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

import cv2
import numpy as np


def count_images(args):
    return count_or_load_images(args, True)


def load_images(args):
    return count_or_load_images(args, False)


# todo: change regex to pathlib.glob
def count_or_load_images(args, count_only):
    path = os.path.abspath(args.input)
    pat = None
    args.files = []
    if not hasattr(args, 're_path') or args.re_path is None:
        args.re_path = None
    else:
        rex = args.re_path
        try:
            pat = re.compile(rex)
        except Exception as err:
            log.error(f"Invalid regex {rex} {err}")
            return 0

    if args.re_path is None:
        if not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            if not count_only:
                if args.verbose > 0:
                    log.debug(f"adding image {path}")
                args.files.append(path)
            return 1
        if not os.path.isdir(path):
            return 0

    count = 0
    limit = args.start + args.count
    idx = args.start
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
                if args.verbose > 1:
                    log.debug(f"adding image {count}/{limit}  {fp}")
                args.files.append(fp)
            continue
        # regex
        m = pat.match(f)
        if m is None:
            continue
        fp = join(path, f)
        count += 1
        idx += 1
        if not count_only:
            if args.verbose > 1:
                log.debug(f"adding image {count}/{limit}  {fp}")
            args.files.append(fp)

    if args.verbose > 1:
        log.debug(f"{count} images")
    return count


def preprocess_images(args):
    result = []
    start = time.perf_counter()
    for file in args.files:
        if Path(file).is_dir():
            continue
        if args.verbose > 1:
            log.debug(f"file {file}")
        result.append(Image.open(file).convert('RGB').resize(args.size, Image.ANTIALIAS))

    duration = (time.perf_counter() - start) / 1000
    if duration > 2:
        log.debug(f"preprocessing took {duration} ms")
    return result


def preprocess_batch(args, idx):
    args.np_images = np.ndarray(shape=(args.batch_size, args.c, args.h, args.w))
    for i in range(idx, idx+args.batch_size):
        if i >= len(args.files[i]):
            return
        file = args.files[i]
        image = cv2.imread(file)
        if image.shape[:-1] != (args.h, args.w):
            log.debug(f"resize from {image.shape[:-1]} to {(args.h, args.w)}")
            image = cv2.resize(image, (args.w, args.h))
        # Change data layout from HWC to CHW
        args.np_images[i-idx] = image.transpose((2, 0, 1))
    return


def copy_to_dir(args, src_file_path, dest_dir_path):
    if not dest_dir_path.exists():
        os.makedirs(dest_dir_path)
    dest = Path(dest_dir_path, src_file_path.name)
    if dest.exists():
        return False
    if args.verbose > 1:
        log.debug(f"Copying {src_file_path.name} to {dest_dir_path}")
    shutil.copy2(src_file_path, dest_dir_path)


def test():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)
    log.debug("testing...")
    from py.common.args import parse_args
    args = parse_args("test")
    args.verbose = 2
    args.input = "../../images"
    args.re_path = R'dog.*\.jpg'
    args.model = "../../models/squeezenet1.1/FP16/squeezenet1.1.xml"
    assert Path(args.input).exists()
    args.count = n = count_images(args)
    assert 0 < n <= 100
    args.start = n - 2
    args.count = 5
    args.n, args.c, args.h, args.w = 1, 3, 227, 227
    args.size = (args.w, args.h)

    load_images(args)
    m = len(args.files)
    assert 0 <= m <= 5

    args.images = preprocess_images(args)
    for idx in range(len(args.images)):
        preprocess_batch(args, idx)
        assert args.n == args.np_images.shape[0]

    log.info("OK")


if __name__ == '__main__':
    test()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %6.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

# files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
# files = files[args.start: args.start + args.count]
