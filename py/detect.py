import logging as log
import os
import sys
import time
from pathlib import Path

import cv2
import ngraph as ng
import numpy as np
from openvino.inference_engine import IECore

from common import util
from common.arg_parser import parse_args

version = "v.2021.1.22"


def main():
    args = init()
    log.info(f"Object detection benchmark {version}")

    util.prepare()

    # # check how many images are available
    # count = util.count_images(args)
    # if count < args.count:
    #     args.count = count
    # # Read and pre-process input images
    # util.load_images(args)
    # if len(args.files) == 0:
    #     log.info(f"empty input set")
    #     exit(0)
    #
    # log.info(f"Loaded {len(args.files)} image(s)")
    # log.info(f"Image preparation")
    # assert Path(args.model).exists()
    # log.info(f"Loading network: {args.model}")
    #
    # # initialize openvino engine
    # args.core = init_engine(args)

    util.preprocess_images(args)

    # Load network model
    network = args.core.load_network(network=args.net, device_name=args.device)

    log.info(f"device {args.device}")
    log.info(f"Starting inference in synchronous mode")
    # accumulates inference time
    inference_duration = 0
    total = 0
    idx = 0
    failed = 0
    log.info(f"START - repeating {args.repeat} time(s)")
    process(args)


def process(args):
    for input_key in args.net.input_info:
        el = args.net.input_info[input_key]

        if len(el.layout) == 4:
            input_name = input_key
            args.net.input_info[input_key].precision = 'U8'
        elif len(args.net.input_info[input_key].layout) == 2:
            input_info_name = input_key
            el.precision = 'FP32'
            if (el.input_data.shape[1] != 3 and
                el.input_data.shape[1] != 6) or \
                    el.input_data.shape[0] != 1:
                log.error('Invalid input info. Should be 3 or 6 values length.')

    data = {input_name: images}
    n = args.count
    c = args.c

    if input_info_name != "":
        infos = np.ndarray(shape=(n, c), dtype=float)
        for i in range(n):
            infos[i, 0] = args.h
            infos[i, 1] = args.w
            infos[i, 2] = 1.0
        data[input_info_name] = infos

    out_blob = next(iter(net.outputs))

    output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
    output_ops = {op.friendly_name: op for op in args.ops
                  if op.friendly_name in net.outputs and op.get_type_name() == "DetectionOutput"}

    if len(output_ops) != 0:
        output_name, output_info = output_ops.popitem()

    if output_name == "":
        log.error("Can't find a DetectionOutput layer in the topology")

    output_dims = output_info.shape
    if len(output_dims) != 4:
        log.error("Incorrect output dimensions for SSD model")
    max_proposal_count, object_size = output_dims[2], output_dims[3]

    if object_size != 7:
        log.error("Output item should have 7 as a last dimension")

    output_info.precision = "FP32"
    min_conf = 0.25
    ###############################
    log.info("performing inference")
    res = network.infer(inputs=data)
    ###############################

    log.info("Processing results")
    res = res[out_blob]
    boxes, classes = {}, {}
    data = res[0][0]
    # draw rectangles over original image
    for number, proposal in enumerate(data):
        if proposal[2] > 0:
            imid = np.int(proposal[0])
            ih, iw = args.images_hw[imid]
            label = np.int(proposal[1])
            # print("imid", imid, "id", proposal[1], "label", label, "iw", iw, "ih", ih)
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            if confidence >= min_conf:
                # print("[{},{}] element, conf = {:.6}    ({},{})-({},{}) batch id : {}"
                #      .format(number, label, confidence, xmin, ymin, xmax, ymax, imid))
                if imid not in boxes.keys():
                    boxes[imid] = []
                boxes[imid].append([xmin, ymin, xmax, ymax])
                if imid not in classes.keys():
                    classes[imid] = []
                classes[imid].append(label)
            else:
                print()

    max_w = 640
    min_w = 600
    for imid in classes:
        result = cv2.imread(args.files[imid])
        for box in boxes[imid]:
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
        if iw > max_w:
            r = max_w / iw
            w = int(r*iw)
            h = int(r*ih)
            result = cv2.resize(result, (w, h))
        else:
            if iw < min_w:
                r = min_w / iw
                w = int(r * iw)
                h = int(r * ih)
                result = cv2.resize(result, (w, h))

        cv2.imwrite("/tmp/out.jpg", result)
        log.info("Image /tmp/out.jpg created!")

        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    args = parse_args("classification")
    assert os.path.exists(args.input)
    args.t0 = time.perf_counter()
    args.count = 10
    args.device = "MYRIAD"
    args.verbose = 0
    return args


def init_engine(args):
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    engine = IECore()
    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    if not os.path.exists(args.model):
        log.error(f"Model {args.model} not found")
        exit(1)

    log.debug(f"Loading model: {args.model}")
    net = engine.read_network(args.model)
    func = ng.function_from_cnn(net)
    args.ops = func.get_ordered_ops()
    args.input_blob = next(iter(net.input_info))
    args.out_blob = next(iter(net.outputs))
    net.batch_size = args.count

    # print("inputs count: " + str(len(net.input_info.keys())))
    for input_key in net.input_info:
        # print("input shape: " + str(net.input_info[input_key].input_data.shape))
        # print("input key: " + input_key)
        el = net.input_info[input_key].input_data
        # print("LAYOUT", el.layout)
        if len(el.layout) == 4:
            args.n, args.c, args.h, args.w = el.shape
            break
    args.net = net
    args.engine = engine
    log.debug(f"Batch size: {args.n}, h {args.h}, w {args.w}, {args.c} channels")
    return


if __name__ == '__main__':
    main()
