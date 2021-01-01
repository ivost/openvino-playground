import time
import logging as log
import os
import sys
import time
from argparse import ArgumentParser, SUPPRESS
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
    print("image count", count)
    return count


def load_images(args):
    path = os.path.abspath(args.input)
    files = [path]
    if os.path.isdir(path):
        print("loading images from", path)
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        #files.sort()
        files = files[args.start: args.start + args.count]
    args.files = files
    images = np.ndarray(shape=(args.count, args.c, args.h, args.w))
    for i in range(args.count):
        image = cv2.imread(files[i])
        if image is None:
            log.error("File {} not found".format(files[i]))
            continue
        if image.shape[:-1] != (args.h, args.w):
            # log.info("Image {} is resized from {} to {}".format(files[i], image.shape[:-1], (h, w)))
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
