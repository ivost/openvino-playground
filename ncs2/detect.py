import cv2
import numpy as np
import logging as log

from ncs2.engine import Engine
from ncs2.imageproc import ImageProc
from ncs2.stats import Stats


class Detect(Engine):

    def __init__(self, log_level=log.INFO):
        super().__init__("Object detection benchmark", "v.2021.1.23")
        # todo: move to super?
        self.args.model = "../models/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.xml"
        self.img_proc = ImageProc(self.args)

    def main(self):
        args = self.args
        self.img_proc.prepare()
        stats = Stats()
        stats.begin()
        # only reading files
        self.img_proc.preprocess_images()
        files = self.img_proc.files
        log.info(f"{len(files)} images")
        log.info(f"repeating {args.repeat} time(s)")
        log.info(f"START")

        net = args.net

        # n = 1 (batch size)
        # todo: loop over N images

        for input_key in net.input_info:
            print("input shape: " + str(net.input_info[input_key].input_data.shape))
            print("input key: " + input_key)
            if len(net.input_info[input_key].input_data.layout) == 4:
                n, c, h, w = net.input_info[input_key].input_data.shape

        images = np.ndarray(shape=(n, c, h, w))
        images_hw = []
        for i in range(n):
            # read with opencv
            file = files[i]
            log.info(f"reading {file}")
            image = cv2.imread(file)
            ih, iw = image.shape[:-1]
            images_hw.append((ih, iw))
            if (ih, iw) != (h, w):
                image = cv2.resize(image, (w, h))
            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            images[i] = image

        data = self.prepare_input(net, images, args)
        if not self.model_check(net, args):
            log.error("Model error")

        ###############################
        log.info("performing inference")
        stats.mark()
        res = args.network.infer(inputs=data)
        stats.bump()
        ###############################
        stats.end()
        log.info(str(stats))
        self.process_results(res, images_hw)

    def prepare_input(self, net, images, args):
        log.info("Preparing input blobs")
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
        n = args.batch_size
        c = args.c

        if input_info_name != "":
            log.info(f"input_info_name {input_info_name}, input_name {input_name}")
            infos = np.ndarray(shape=(n, c), dtype=float)
            for i in range(n):
                infos[i, 0] = args.h
                infos[i, 1] = args.w
                infos[i, 2] = 1.0
            data[input_info_name] = infos
        return data

    def process_results(self, res, images_hw):
        min_conf = 0.25
        out_blob = self.img_proc.args.out_blob
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

        self.show_result(classes, boxes, ih, iw)

    def show_result(self, classes, boxes, ih, iw):
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

    def model_check(self, net, args):
        log.info('Preparing output blobs')
        output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
        output_ops = {op.friendly_name: op for op in args.ops
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

if __name__ == '__main__':
    d = Detect()
    d.main()
