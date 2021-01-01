#!/usr/bin/env python3
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 INP=$D/images/cat01.jpeg
INP=$D/images/dog03.jpeg

MODEL=$D/models/ir/public/squeezenet1.1/FP16/squeezenet1.1

"""
from __future__ import print_function

import logging as log
import os
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')

    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      # default="images",
                      default=os.environ['HOME'] + "/data/imagen",
                      # default="images/cat05.jpg",
                      # default="images/cat09.jpg",
                      # default="images/duo.jpg",
                      # default="images/duo10.jpg",
                      type=str)
    args.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.",
                      default="models/ir/public/squeezenet1.1/FP16/squeezenet1.1.xml",
                      type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file",
                      default="models/ir/public/squeezenet1.1/FP16/squeezenet1.1.labels",
                      type=str)

    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: "
                           "is acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("-s", "--start", help="Optional. Start index (when directory)", default=0, type=int)
    args.add_argument("-n", "--number", help="Optional. Max number of images to process", default=10, type=int)
    args.add_argument("-q", "--quiet", help="Optional. Quiet mode - don't write to the output", default=False,
                      type=bool)
    args.add_argument("-tn", "--top", help="Optional. Number of top results", default=3, type=int)
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    # os.chdir(os.path.dirname(__file__))
    print("Working directory", os.getcwd())

    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    model = args.model
    log.info(f"Loading network: {model}")
    net = ie.read_network(model=model)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    path = os.path.abspath(args.input)
    files = [path]
    if os.path.isdir(path):
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        files.sort()
        files = files[args.start: args.start + args.number]

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(files)

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    log.info("Batch size: {}".format(n))
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(files[i])
        if image is None:
            log.error("File {} not found".format(files[i]))
            continue
        if image.shape[:-1] != (h, w):
            # log.info("Image {} is resized from {} to {}".format(files[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image

    # Loading model to the plugin
    # log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    #do_process(exec_net)

    start_time = time.time()
    res = exec_net.infer(inputs={input_blob: images})
    if not args.quiet:
        show_results(args, res, files, n, out_blob)
    elapsed_time = time.time() - start_time
    # print("elapsed time", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    log.info("elapsed time: {:.3} sec".format(elapsed_time))
    log.info("     average: {:.3} ms".format(1000 * elapsed_time / n))

# todo
# @timeit
# def do_process(**kwargs):
#         name = kw.get('log_name', method.__name__.upper())
#    return exec_net.infer(inputs={input_blob: images})


def show_results(args, res, files, n, out_blob):
    min_prob = 0.2
    # Processing output blob
    # log.info("Processing output blob")
    res = res[out_blob]
    # log.info("Top {} results: ".format(args.top))

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None

    # classid_str = "class"
    # probability_str = "prob."
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.top:][::-1]
        print("Image {}/{} - {}".format(i + 1, n, files[i]))
        # print(probability_str, classid_str)
        # print("{}  {}".format('-' * len(probability_str), '-' * len(classid_str)))
        for id in top_ind:
            if probs[id] < min_prob:
                break
            det_label = labels_map[id] if labels_map else "{}".format(id)
            print("{:.2f} {}".format(probs[id], det_label))


if __name__ == '__main__':
    main()
