import re
import shutil
import sys
import time
from os import listdir
from os.path import isfile, join
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


def preproces_images(args):
    result = []

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        file = args.input
        log.info(f"input {file}")
        if not Path(file).exists():
            log.error(f"{file} NOT FOUND ")
            exit(4)

        image = cv2.imread(file)
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Batch size is {}".format(n))

    # start = time.perf_counter()
    # for file in args.files:
    #     if Path(file).is_dir():
    #         continue
    #     if args.verbose > 1:
    #         log.debug(f"file {file}")
    #     result.append(Image.open(file).convert('RGB').resize(args.size, Image.ANTIALIAS))
    #
    # duration = (time.perf_counter() - start) / 1000
    # if duration > 2:
    #     log.debug(f"preprocessing took {duration} ms")
    return result


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

    # log.debug("testing...")

    from py.common.args import parse_args
    args = parse_args("test")
    args.input = "../../images"

    n = count_images(args)
    assert 0 <= n <= 100

    args.start = n - 2
    args.count = 5
    load_images(args)
    m = len(args.files)
    assert 0 <= m <= 5

    # must use raw string and valid regex "cat*.jpg" -> "cat.*\.jpg"
    # todo: fixme
    # args.re_path = R'dog.*\.jpg'
    # args.count = 0
    # args.number = 3
    # args.verbose = 2
    # m = count_images(args)
    # load_images(args)
    # m = len(args.files)
    # assert 0 < m <= 3


if __name__ == '__main__':
    test()

# def count_images(args):
#     path = os.path.abspath(args.input)
#     if not os.path.exists(path):
#         return 0
#     if os.path.isfile(path):
#         return 1
#     if not os.path.isdir(path):
#         return 0
#     count = 0
#     max = args.start + args.number
#     for f in listdir(path):
#         if count >= max:
#             break
#         if isfile(join(path, f)):
#             count += 1
#     return count
#
#
# def load_images(args):
#     path = os.path.abspath(args.input)
#     files = []
#     count = 0
#     if os.path.isdir(path):
#         log.debug(f"loading images from {path}")
#         for f in listdir(path):
#             if count == args.count:
#                 break
#             if isfile(join(path, f)):
#                 count += 1
#                 files.append(join(path, f))
#     else:
#         log.debug(f"loading image {path}")
#         files.append(path)
#         count = 1
#     assert args.count == count
#     args.files = files
#
#     images = np.ndarray(shape=(args.count, args.c, args.h, args.w))
#     args.images_hw = []
#     for i in range(args.count):
#         # print(files[i])
#         image = cv2.imread(files[i])
#         if image is None:
#             log.error("File {} {} not found".format(i, files[i]))
#             continue
#         ih, iw = image.shape[:-1]
#         args.images_hw.append((ih, iw))
#         if image.shape[:-1] != (args.h, args.w):
#             # log.debug("Image {} is resized from {} to {}".format(files[i], image.shape[:-1], (args.h, args.w)))
#             image = cv2.resize(image, (args.w, args.h))
#         image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
#         images[i] = image
#     return images


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

        #files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        #files = files[args.start: args.start + args.count]
