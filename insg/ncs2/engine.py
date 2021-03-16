import logging as log
import sys

import cv2
import ngraph as ng
import numpy as np
from openvino.inference_engine import IECore

from insg.common.config import Config
from insg.common.imageproc import ImageProc


class Engine:

    def __init__(self, message, version, config_ini='config.ini', log_level=log.INFO):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        log.info(f"\n{message} {version}\n")

        self.c = Config()
        self.c.read(config_ini)

        n = self.c.network

        # self.model = Config.existing_path(n.model)
        # self.weights = Config.existing_path(n.weights)
        model = Config.existing_path(n.model)
        self.input = Config.existing_path(self.c.input.images)
        labels = Config.existing_path(n.labels)

        with open(labels, 'r') as file:
            self.labels = [line.split(sep=' ', maxsplit=1)[-1].strip() for line in file]
            log.debug(f"{len(self.labels)} labels")

        # initialize openvino engine
        log.info(f"Initializing OpenVINO Inference Engine, device {n.device}")
        self.core = IECore()
        # Plugin initialization for specified device and load extensions library if specified
        # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        log.info(f"Loading model: {model}")
        net = self.core.read_network(model=model)
        self.network = self.core.load_network(network=net, device_name=n.device)
        assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        func = ng.function_from_cnn(net)
        self.ops = func.get_ordered_ops()

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
        files = self.img_proc.files
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-top:][::-1]
            if verbose > 0:
                print("\nImage {}/{} - {}".format(idx + 1, len(files), files[idx]))
            count = 0
            for id in top_ind:
                print("id", id, "probs[id]", probs[id])
                if probs[id] < min_prob:
                    break
                label = str(id)
                if self.labels and id < len(self.labels):
                    label = self.labels[id]
                if verbose > 0:
                    print("{:4.1%} {} [{}]".format(probs[id], label, id))
                count += 1
            if count == 0:
                if verbose > 0:
                    print("--")
            return count > 0

    def process_detection_results(self, res, img_file, images_hw):
        out_blob = self.out_blob
        res = res[out_blob]
        boxes, classes = {}, {}
        data = res[0][0]

        min_conf = float(self.c.network.confidence)
        top = int(self.c.network.top)
        verbose = int(self.c.output.verbose)

        ih = iw = 0
        # draw rectangles over original image
        for count, proposal in enumerate(data):
            conf = proposal[2]
            if conf < min_conf:
                continue
            imid = np.int(proposal[0])
            ih, iw = images_hw[imid]
            idx = int(np.int(proposal[1]))
            if self.labels and 0 < idx <= len(self.labels):
                label = self.labels[idx-1]
            else:
                label = str(idx)
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            if imid not in boxes.keys():
                boxes[imid] = []
            boxes[imid].append([xmin, ymin, xmax, ymax])
            if imid not in classes.keys():
                classes[imid] = []
            classes[imid].append(label)
            log.debug(f"conf {conf}, label {label}")
            count += 1
            if count > top:
                break

        if ih > 0 and iw > 0:
            self.display_result(img_file, classes, boxes, ih, iw)

    def display_result(self, img_file, classes, boxes, ih, iw):
        max_w = 640
        min_w = 600
        # color = (232, 35, 244)
        color = (0, 240, 240)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = color
        line_type = 2
        for imid in classes:
            image = cv2.imread(img_file)
            idx = 0
            for box in boxes[imid]:
                text = classes[imid][idx]
                idx += 1
                x, y = box[0], box[1]
                x1 = x + 10 if x < 10 else x - 10
                y1 = y + 20 if y < 20 else y - 20
                cv2.rectangle(image, (x, y), (box[2], box[3]), color, line_type)
                cv2.putText(image, text, (x1, y1), font, font_scale, font_color, line_type)

            if iw > max_w:
                r = max_w / iw
                w = int(r*iw)
                h = int(r*ih)
                image = cv2.resize(image, (w, h))
            else:
                if iw < min_w:
                    r = min_w / iw
                    w = int(r * iw)
                    h = int(r * ih)
                    image = cv2.resize(image, (w, h))

            cv2.imwrite("/tmp/out.jpg", image)
            log.info("Image /tmp/out.jpg created!")

            cv2.imshow("result", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return


if __name__ == '__main__':
    engine = Engine("init", "engine", "2021.3.16")
