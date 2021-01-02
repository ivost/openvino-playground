import logging as log
import os
import sys
from argparse import ArgumentParser, SUPPRESS

import cv2
import ngraph as ng
import numpy as np
from openvino.inference_engine import IECore

from python.common import util
from python.common.args import parse_args


def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    args = parse_args("classification")
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
    # assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    # assert len(net.outputs) == 1, "Sample supports only single output topologies"
    args.input_blob = next(iter(net.input_info))
    args.out_blob = next(iter(net.outputs))
    net.batch_size = args.count
    args.n, args.c, args.h, args.w = net.input_info[args.input_blob].input_data.shape
    log.debug(f"Batch size: {args.n}, h {args.h}, w {args.w}, {args.c} channels")
    args.net = net
    return engine


def main():
    args = init()
    # check how many images are available
    args.count = util.count_images(args)
    # initialize openvino engine
    engine = init_engine(args)

    # Read and pre-process input images
    images = util.load_images(args)
    log.info(f"{len(images)} image(s)")

    # Load network model
    net = engine.load_network(args.net, args.device)
    log.info("Preparing input blobs")
    assert (len(net.input_info.keys()) == 1 or len(
        net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    out_blob = next(iter(net.outputs))
    input_name, input_info_name = "", ""

    for input_key in net.input_info:
        el = net.input_info[input_key]
        if len(el.layout) == 4:
            input_name = input_key
            log.info("Batch size is {}".format(net.batch_size))
            el.precision = 'U8'

        elif len(el.layout) == 2:
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

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

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
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Loading model to the device")
    network = engine.load_network(args.net, args.device)

    log.info("Creating infer request and starting inference")
    res = network.infer(inputs=data)
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info("Processing output blobs")
    res = res[out_blob]
    boxes, classes = {}, {}
    data = res[0][0]
    for number, proposal in enumerate(data):
        if proposal[2] > 0:
            imid = np.int(proposal[0])
            ih, iw = args.images_hw[imid]
            label = np.int(proposal[1])
            print("id", proposal[1], "label", label)
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}" \
                  .format(number, label, confidence, xmin, ymin, xmax, ymax, imid), end="")
            if proposal[2] > 0.5:
                print(" WILL BE PRINTED!")
                if not imid in boxes.keys():
                    boxes[imid] = []
                boxes[imid].append([xmin, ymin, xmax, ymax])
                if not imid in classes.keys():
                    classes[imid] = []
                classes[imid].append(label)
            else:
                print()

    for imid in classes:
        result = cv2.imread(args.input[imid])
        for box in boxes[imid]:
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)

        # w = 320
        # h = 240
        # result = cv2.resize(result, (w, h))

        cv2.imwrite("out.bmp", result)
        log.info("Image out.bmp created!")

        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
