import logging as log

from insg.common import Stats
from insg.ncs2.engine import Engine

version = "v.2021.3.16"


class Classify(Engine):
    # todo cmd line arg to choose ini file
    def __init__(self, log_level=log.INFO):
        super().__init__("Classification benchmark", version, "classify.ini")

    def main(self):
        stats = Stats()
        repeat = int(self.c.input.repeat)
        stats.begin()
        img_proc = self.img_proc
        img_proc.preprocess_images(self.size)
        log.info(f"{len(self.img_proc.files)} images")
        log.info(f"repeating {repeat} time(s)")
        for _ in range(repeat):
            print(".", end="", flush=True)
            # assuming batch size = 1
            for idx in range(len(self.img_proc.files)):
                images, images_hw = self.img_proc.preprocess_batch(idx, self.batch_size, self.channels, self.height, self.width)
                ###############################
                # inference
                stats.mark()
                res = self.network.infer(inputs={self.input_blob: images})
                failed = not self.process_classification_results(res, idx)
                stats.bump(failed)
                ###############################
        stats.end()
        print("", flush=True)
        log.info(stats.summary())


if __name__ == '__main__':
    c = Classify()
    c.main()
