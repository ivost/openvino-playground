import logging as log

from depthai import Engine, Stats


class Detect(Engine):

    def __init__(self, log_level=log.DEBUG):
        super().__init__("Object detection benchmark", "v.2021.1.25", "detect.ini", log_level=log_level)

    def main(self):
        stats = Stats()
        return

        # repeat = int(self.c.input.repeat)
        # stats.begin()
        # img_proc = self.img_proc
        # img_proc.preprocess_images(self.size)
        # log.info(f"{len(img_proc.files)} images")
        # log.info(f"repeating {repeat} time(s)")
        # for _ in range(repeat):
        #     print(".", end="", flush=True)
        #     for idx in range(len(self.img_proc.files)):
        #         images, images_hw = self.img_proc.preprocess_batch(idx, self.batch_size, self.channels, self.height,
        #                                                            self.width)
        #         data = self.prepare_input(images)
        #         ###############################
        #         # inference
        #         stats.mark()
        #         res = self.network.infer(inputs=data)
        #         stats.bump()
        #         ###############################
        #         self.process_detection_results(res, self.img_proc.files[idx], images_hw)
        # stats.end()
        # print("", flush=True)
        # log.info(stats.summary())


if __name__ == '__main__':
    d = Detect()
    d.main()
