
import logging as log
import time

import numpy as np

from common import util

version = "v.2021.1.22"

class Classify:

    def __init__(self):


    def main():
        args = util.init()
        log.info(f"Classification benchmark {version}")
        util.prepare(args)
        log.info(f"Starting inference in synchronous mode")
        log.info(f"START - repeating {args.repeat} time(s)")

        for _ in range(args.repeat):
            # assuming batch size = 1
            for idx in range(len(args.files)):
                util.preprocess_batch(args, idx)
                t1 = time.perf_counter()
                # inference
                res = args.network.infer(inputs={args.input_blob: args.np_images})
                args.inference_duration += time.perf_counter() - t1
                if not check_results(args, res, idx):
                    args.failed += 1
                args.total += 1
        util.show_starts(args)
        log.info(f"  END")




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
            if args.verbose > 0:
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