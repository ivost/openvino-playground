import logging as log
import sys
import os
from ncs2.config import Config

import numpy as np

from ncs2.stats import Stats
from ncs2.imageproc import ImageProc

version = "v.2021.1.23"


class Classify:

    def __init__(self, log_level=log.INFO):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        config = Config()
        args = config.parse()
        if not os.path.exists(args.input):
            log.error(f"{args.input} not found")
            exit(0)
        args.verbose = 0
        self.args = args

    def main(self):
        config = self.args
        img_proc = ImageProc(config)
        stats = Stats()
        img_proc.prepare()
        stats.begin()
        img_proc.preprocess_images()
        log.info(f"Classification benchmark {version}")
        log.info(f"Starting inference in synchronous mode")
        log.info(f"{len(img_proc.files)} images")
        log.info(f"repeating {config.repeat} time(s)")
        log.info(f"START")

        for _ in range(config.repeat):
            # if config.verbose > 0:
            print(".", end="", flush=True)
            # assuming batch size = 1
            for idx in range(len(img_proc.files)):
                img_proc.preprocess_batch(idx)
                # inference
                stats.mark()
                res = config.network.infer(inputs={config.input_blob: config.np_images})
                failed = not self.check_results(res, idx)
                stats.bump(failed)
        stats.end()
        print("")
        log.info(str(stats))
        log.info(f"END")

    def check_results(self, result, idx):
        args = self.args
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
            if args.verbose > 0:
                print("\nImage {}/{} - {}".format(idx + 1, len(args.files), args.files[idx]))
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
    c = Classify()
    c.main()
