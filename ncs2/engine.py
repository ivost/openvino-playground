import logging as log
import sys

import cv2
import ngraph as ng
import numpy as np
from openvino.inference_engine import IECore

from ncs2.config import Config
from ncs2.imageproc import ImageProc


class Engine:

    def __init__(self, message, version, config_ini, log_level=log.INFO):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)

        self.c = Config()
        self.c.read(config_ini)
        # print(self.c)
        n = self.c.network
        self.model = Config.existing_path(n.model)
        self.weights = Config.existing_path(n.weights)
        self.model = Config.existing_path(n.model)
        self.labels = Config.existing_path(n.labels)

        self.input = Config.existing_path(self.c.input.images)
        # self.c.verbose = 1
        log.info(f"{message} {version}")
        log.info(f"Creating OpenVINO Inference Engine, device {n.device}")

        # initialize openvino engine
        self.core = IECore()
        # Plugin initialization for specified device and load extensions library if specified
        # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        log.info(f"Loading model: {self.model}")
        net = self.core.read_network(model=self.model)
        self.network = self.core.load_network(network=net, device_name=n.device)
        assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        func = ng.function_from_cnn(net)
        self.ops = func.get_ordered_ops()
        # todo: refactor with class?
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        self.batch_size, self.channels, self.height, self.width = net.input_info[self.input_blob].input_data.shape
        self.size = (self.width, self.height)
        self.net = net

        self.img_proc = ImageProc(self.c)
        self.img_proc.prepare()
        return

    def prepare_input(self, images):
        net = self.net
        # log.("Preparing input blobs")
        input_name, input_info_name = "", ""
        for input_key in net.input_info:
            el = net.input_info[input_key]
            if len(el.layout) == 4:
                input_name = input_key
                net.input_info[input_key].precision = 'U8'
            elif len(net.input_info[input_key].layout) == 2:
                input_info_name = input_key
                el.precision = 'FP32'
                if (el.input_data.shape[1] != 3 and
                    el.input_data.shape[1] != 6) or \
                        el.input_data.shape[0] != 1:
                    log.error('Invalid input info. Should be 3 or 6 values length.')
        data = {input_name: images}
        if input_info_name != "":
            log.info(f"input_info_name {input_info_name}, input_name {input_name}")
            infos = np.ndarray(shape=(self.batch_size, self.channels), dtype=float)
            for i in range(self.batch_size):
                infos[i, 0] = self.height
                infos[i, 1] = self.width
                infos[i, 2] = 1.0
            data[input_info_name] = infos
        return data

    def model_check(self):
        net = self.net
        # log.info('Preparing output blobs')
        output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
        output_ops = {op.friendly_name: op for op in self.ops
                      if op.friendly_name in net.outputs and op.get_type_name() == "DetectionOutput"}

        if len(output_ops) != 0:
            output_name, output_info = output_ops.popitem()

        if output_name == "":
            log.error("Can't find a DetectionOutput layer in the topology")
            return False

        output_dims = output_info.shape
        if len(output_dims) != 4:
            log.error("Incorrect output dimensions for SSD model")
            return False

        max_proposal_count, object_size = output_dims[2], output_dims[3]

        if object_size != 7:
            log.error("Output item should have 7 as a last dimension")

        output_info.precision = "FP32"
        return True

    def process_classification_results(self, result, idx):
        # from config
        min_prob = float(self.c.network.confidence)
        top = int(self.c.network.top)
        res = result[self.out_blob]
        verbose = int(self.c.output.verbose)

        # todo: cache early
        # if self.labels:
        #     with open(self.labels, 'r') as f:
        #         labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        # else:
        #     return

        with open(self.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]

        files = self.img_proc.files
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-top:][::-1]
            if verbose > 0:
                print("\nImage {}/{} - {}".format(idx + 1, len(files), files[idx]))
            count = 0
            for id in top_ind:
                if probs[id] < min_prob:
                    break
                label = labels_map[id] if labels_map else "{}".format(id)
                if verbose > 0:
                    print("{:4.1%} {} [{}]".format(probs[id], label, id))
                count += 1
            if count == 0:
                if verbose > 0:
                    print("--")
            return count > 0

    def process_detection_results(self, res, images_hw):
        min_conf = 0.25
        out_blob = self.out_blob
        log.info(f"Processing results {out_blob}")
        res = res[out_blob]
        boxes, classes = {}, {}
        data = res[0][0]
        # draw rectangles over original image
        for number, proposal in enumerate(data):
            if proposal[2] > 0:
                imid = np.int(proposal[0])
                ih, iw = images_hw[imid]
                label = np.int(proposal[1])
                # print("imid", imid, "id", proposal[1], "label", label, "iw", iw, "ih", ih)
                confidence = proposal[2]
                xmin = np.int(iw * proposal[3])
                ymin = np.int(ih * proposal[4])
                xmax = np.int(iw * proposal[5])
                ymax = np.int(ih * proposal[6])
                if confidence >= min_conf:
                    # print("[{},{}] element, conf = {:.6}    ({},{})-({},{}) batch id : {}".format(number, label, confidence, xmin, ymin, xmax, ymax, imid))
                    if imid not in boxes.keys():
                        boxes[imid] = []
                    boxes[imid].append([xmin, ymin, xmax, ymax])
                    if imid not in classes.keys():
                        classes[imid] = []
                    classes[imid].append(label)
                else:
                    print()

        # self.display_result(classes, boxes, ih, iw)

    def display_result(self, classes, boxes, ih, iw):
        files = self.img_proc.files
        max_w = 640
        min_w = 600
        for imid in classes:
            result = cv2.imread(files[imid])
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
        return


if __name__ == '__main__':
    engine = Engine("init", "engine")
