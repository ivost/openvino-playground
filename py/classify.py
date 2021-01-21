
import logging as log
import os
import sys
import time
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore

from py.common import util
from py.common.args import parse_args

version = "v.2021.1.20"


def main():
    t0 = time.perf_counter()
    args = init()
    log.info(f"Classification benchmark {version}")

    args = init()
    assert os.path.exists(args.input)
    args.count = 100
    # check how many images are available
    count = util.count_images(args)
    if count < args.count:
        args.count = count

    args.verbose = 0

    # Read and pre-process input images
    images = util.load_images(args)
    if len(args.files) == 0:
        log.info(f"empty input set")
        exit(0)

    log.info(f"Loaded {len(args.files)} image(s)")

    log.info(f"Image preparation")

    assert Path(args.model).exists()
    log.info(f"Loading network: {args.model}")
    from openvino.inference_engine import IECore
    ie = IECore()
    net = ie.read_network(model=args.model)

    # initialize openvino engine
    engine = init_engine(args)

    exec_net = ie.load_network(network=args.net, device_name=args.device)

    images = util.preprocess_images(args)

    # Load network model
    network = engine.load_network(args.net, args.device)

    log.info("device {args.device}")
    log.info("Starting inference in synchronous mode")
    # accumulates inference time
    inference_duration = 0
    total = 0
    idx = 0
    failed = 0
    # todo: args
    args.repeat = 2
    log.info(f"START - repeating {args.repeat} time(s)")

    for _ in range(args.repeat):
        # assuming batch size = 1
        for idx in range(len(args.files)):
            util.preprocess_batch(args, idx)
            t1 = time.perf_counter()
            # inference
            res = exec_net.infer(inputs={args.input_blob: args.np_images})
            inference_duration += time.perf_counter() - t1
            if not check_results(args, res, idx):
                failed += 1
            total += 1

    dur = time.perf_counter() - t0
    avg = (inference_duration * 1000) / total
    log.info(f"  Total images: {total}, not classified: {failed}")
    log.info(f"Inference time: {inference_duration*1000:.0f} ms")
    log.info(f"       Average: {avg:.2f} ms")
    log.info(f"  Elapsed time: {dur*1000:.0f} ms")
    # if out_dir:
    #     log.info(f"Results are in {out_dir}")
    log.info(f"  END")


def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = parse_args("classification")
    return args


def init_engine(args):
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    engine = IECore()
    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    log.debug(f"Loading network: {args.model}")
    net = engine.read_network(args.model)
    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    args.input_blob = next(iter(net.input_info))
    args.out_blob = next(iter(net.outputs))
    args.batch_size, args.c, args.h, args.w = net.input_info[args.input_blob].input_data.shape
    log.debug(f"h {args.h}, w {args.w}")
    args.size = (args.w, args.h)
    args.net = net
    return engine


def check_results(args, result, idx):
    # todo: add arg
    min_prob = 0.25
    res = result[args.out_blob]

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
        return

    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.top:][::-1]
        print("\nImage {}/{} - {}".format(idx+1, len(args.files), args.files[idx]))
        count = 0
        for id in top_ind:
            if probs[id] < min_prob:
                break
            label = labels_map[id] if labels_map else "{}".format(id)
            if args.verbose > 0:
                print("{:4.1%} {} [{}]".format(probs[id], label, id))
            count += 1
        if count == 0:
            if args.verbose > 0:
                print("--")

        return count > 0


if __name__ == '__main__':
    main()