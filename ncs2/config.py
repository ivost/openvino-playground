from argparse import ArgumentParser


class Config:

    def __init(self):
        # log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
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
                            default="MYRIAD",
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
                            default=4,
                            type=int)
        parser.add_argument("-v", "--verbose",
                            help="Optional verbosity level. Used for debugging",
                            type=int,
                            default=0)

        return parser.parse_args()


if __name__ == '__main__':
    import configparser
    import collections

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read('config.ini')
    # print(config.sections)

    for name in config.sections():
        # print(name)
        section = config[name]
        dict = {k: section[k] for k in section}
        # print(f"dict {dict}")
        tup = collections.namedtuple(name, dict)
        setattr(config, name, tup(**dict))

    print(f"config['model']['weights']: {config['model']['weights']}")
    print(f"model.weights: {config.model.weights}")
    print(f"output.dir: {config.output.dir}")

    # section_name = "input"
    # section = config[section_name]
    # InputTuple = collections.namedtuple(section_name, dict)
    # input = InputTuple(**dict)
    # print(f"images: {input.images}")
    # print(f"weights: {input.weights}")
    # print(f"display: {input.display}")
    # config.input = input


    # ap = Config()
    # args = ap.parse()
    # assert args.top == 3
