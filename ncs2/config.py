from argparse import ArgumentParser


class Config:

    def __init(self):
        pass

    def parse(self):
        parser = ArgumentParser()

        parser.add_argument("-i", "--input", help="Required. Path to a image or folder with images.",
                            default="../images",
                            type=str)
        # todo: use Path.glob
        parser.add_argument("-r", "--re_path", help="Optional.",
                            default=None,   # default=R'dog.*\.jpg',
                            type=str)

        parser.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.",
                            default="../models/squeezenet1.1/FP16/squeezenet1.1.xml",
                            type=str)

        parser.add_argument("-l", "--labels", help="Optional. Path to a labels mapping file",
                            default="../models/squeezenet1.1/FP16/squeezenet1.1.labels",
                            type=str)

        parser.add_argument("-d", "--device",
                            help="Optional. Target device: MYRIAD, CPU, GPU, FPGA, HDDL, or HETERO.",
                            default="CPU",
                            type=str)

        parser.add_argument("-o", "--output", help="Optional. Path to output directory.",
                            default=None,
                            type=str)
        parser.add_argument("-s", "--start",
                            help="Optional. Start index (when directory)",
                            default=0, type=int)
        parser.add_argument("-n", "--count",
                            help="Optional. Max number of images to process",
                            default=10, type=int)
        parser.add_argument("-c", "--confidence",
                            help="Optional. Min confidence",
                            default=0.4, type=float)
        parser.add_argument("-T", "--threshold",
                            help="Optional. Threshold",
                            default=0.4, type=float)
        parser.add_argument("-q", "--quiet",
                            help="Optional. If specified will show only perf",
                            action='store_true',
                            default=False)
        parser.add_argument("-t", "--top", help="Optional. Number of top results",
                            default=3,
                            type=int)
        parser.add_argument("--repeat", help="Number of loops over input",
                            default=1,
                            type=int)
        parser.add_argument("-v", "--verbose",
                            help="Optional verbosity level. Use for debugging",
                            type=int,
                            default=2)

        return parser.parse_args()


if __name__ == '__main__':
    ap = Config()
    args = ap.parse()
    assert args.top == 3
