import logging as log
import time

import numpy as np

from ncs2.util import Util

version = "v.2021.1.23"


class Classify:

    def __init__(self):
        self.util = Util()

    def main(self):
        ctx = self.util
        ctx.prepare()
        args = ctx.args
        ctx.preprocess_images()
        log.info(f"Classification benchmark {version}")
        log.info(f"Starting inference in synchronous mode")
        log.info(f"{len(ctx.files)} images")
        log.info(f"repeating {args.repeat} time(s)")
        log.info(f"START")

        for _ in range(args.repeat):
            # assuming batch size = 1
            for idx in range(len(ctx.files)):
                ctx.preprocess_batch(idx)
                t1 = time.perf_counter()
                # inference
                res = args.network.infer(inputs={args.input_blob: args.np_images})
                args.inference_duration += time.perf_counter() - t1
                if not self.check_results(res, idx):
                    args.failed += 1
                args.total += 1
        ctx.show_stats()
        log.info(f"END")

    def check_results(self, result, idx):
        args = self.util.args
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
