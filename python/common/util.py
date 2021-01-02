import logging as log
import os
import time
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np


def count_images(args):
    path = os.path.abspath(args.input)
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return 1
    if not os.path.isdir(path):
        return 0
    count = 0
    max = args.start + args.number
    for f in listdir(path):
        if count >= max:
            break
        if isfile(join(path, f)):
            count += 1
    return count


def load_images(args):
    path = os.path.abspath(args.input)
    files = []
    count = 0
    if os.path.isdir(path):
        log.debug(f"loading images from {path}")
        for f in listdir(path):
            if count == args.count:
                break
            if isfile(join(path, f)):
                count += 1
                files.append(join(path, f))
    else:
        log.debug(f"loading image {path}")
        files.append(path)
        count = 1
    assert args.count == count
    args.files = files

    images = np.ndarray(shape=(args.count, args.c, args.h, args.w))
    args.images_hw = []
    for i in range(args.count):
        # print(files[i])
        image = cv2.imread(files[i])
        if image is None:
            log.error("File {} {} not found".format(i, files[i]))
            continue
        ih, iw = image.shape[:-1]
        args.images_hw.append((ih, iw))
        if image.shape[:-1] != (args.h, args.w):
            # log.debug("Image {} is resized from {} to {}".format(files[i], image.shape[:-1], (args.h, args.w)))
            image = cv2.resize(image, (args.w, args.h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    return images


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
