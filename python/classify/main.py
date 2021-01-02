
import logging as log
import sys
import time

import numpy as np
from openvino.inference_engine import IECore

from python.classify.args import parse_args
from python.common import util


def main():
    args = init()
    # check how many images are available
    args.count = util.count_images(args)
    # initialize openvino engine
    engine = init_engine(args)

    # Read and pre-process input images
    images = util.load_images(args)

    # Loading model to the plugin
    network = engine.load_network(args.net, args.device)

    log.info("Starting inference in synchronous mode")
    start_time = time.time()
    # inference
    res = network.infer(inputs={args.input_blob: images})
    elapsed_time = time.time() - start_time
    if not args.quiet:
        show_results(args, res)
        log.info("elapsed time: {:.3} sec".format(elapsed_time))
        log.info("     average: {:.3} ms".format(1000 * elapsed_time / args.count))


def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = parse_args("classification")
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    return args


def init_engine(args):
    engine = IECore()
    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    log.debug(f"Loading network: {args.model}")
    net = engine.read_network(args.model)
    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    args.input_blob = next(iter(net.input_info))
    args.out_blob = next(iter(net.outputs))
    net.batch_size = args.count
    n, args.c, args.h, args.w = net.input_info[args.input_blob].input_data.shape
    log.debug("Batch size: {}".format(n))
    args.net = net
    return engine


def show_results(args, result):
    min_prob = 0.1
    result = result[args.out_blob]

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None

    for i, probs in enumerate(result):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.top:][::-1]
        print("Image {}/{} - {}".format(i + 1, args.count, args.files[i]))
        for id in top_ind:
            if probs[id] < min_prob:
                break
            det_label = labels_map[id] if labels_map else "{}".format(id)
            print("{:.2f} {}".format(probs[id], det_label))


if __name__ == '__main__':
    main()

