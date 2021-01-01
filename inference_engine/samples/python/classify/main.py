
import logging as log
import sys
import time

import numpy as np
from openvino.inference_engine import IECore

from inference_engine.samples.python.classify.args import build_argparser
from inference_engine.samples.python.common import util


def main():
    args = parse_args()
    args.count = util.count_images(args)

    ie = init_engine(args)

    # Read and pre-process input images
    images = util.load_images(args)

    # Loading model to the plugin
    network = ie.load_network(args.net, args.device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    start_time = time.time()
    res = network.infer(inputs={args.input_blob: images})
    elapsed_time = time.time() - start_time
    log.info("elapsed time: {:.3} sec".format(elapsed_time))
    log.info("     average: {:.3} ms".format(1000 * elapsed_time / args.count))
    if not args.quiet:
        show_results(args, res, args.count, args.out_blob)


def init_engine(args):
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    log.info(f"Loading network: {args.model}")
    net = ie.read_network(args.model)
    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    args.input_blob = next(iter(net.input_info))
    args.out_blob = next(iter(net.outputs))
    net.batch_size = args.count
    n, c, h, w = net.input_info[args.input_blob].input_data.shape
    log.info("Batch size: {}".format(n))
    args.c = c
    args.h = h
    args.w = w
    args.net = net
    return ie


def parse_args():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser()
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    return args


def show_results(args, res, n, out_blob):
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

    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.top:][::-1]
        print("Image {}/{} - {}".format(i + 1, n, args.files[i]))
        for id in top_ind:
            if probs[id] < min_prob:
                break
            det_label = labels_map[id] if labels_map else "{}".format(id)
            print("{:.2f} {}".format(probs[id], det_label))


if __name__ == '__main__':
    main()

