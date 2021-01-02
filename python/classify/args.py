import os

from envargparse import EnvArgParser, EnvArgDefaultsHelpFormatter


def parse_args(name):
    parser = EnvArgParser(prog=name, formatter_class=EnvArgDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", help="Required. Path to a image or folder with images.",
                        env_var="INPUT",
                        default="./images",
                        type=str)
    parser.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.",
                        env_var="MODEL",
                        default="./models/squeezenet1.1/FP16/squeezenet1.1.xml",
                        type=str)
    parser.add_argument("--labels", help="Optional. Path to a labels mapping file",
                        env_var="LABELS",
                        default="./models/squeezenet1.1/FP16/squeezenet1.1.labels",
                        type=str)
    parser.add_argument("-d", "--device",
                        env_var="DEVICE",
                        help="Optional. Target device: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO.",
                        default="CPU", type=str)
    parser.add_argument("-s", "--start",
                        help="Optional. Start index (when directory)",
                        default=0, type=int)
    parser.add_argument("-n", "--number",
                        help="Optional. Max number of images to process",
                        default=10, type=int)
    parser.add_argument("-q", "--quiet",
                        help="Optional. If specified will show only perf",
                        action='store_true',
                        default=False)
    parser.add_argument("-tn", "--top", help="Optional. Number of top results", default=3, type=int)
    #parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args = parser.parse_args()
    if args.number > 100:
        args.number = 100
    print("quiet", args.quiet)
    return args

# todo
# @timeit
# def do_process(**kwargs):
#         name = kw.get('log_name', method.__name__.upper())
#    return exec_net.infer(inputs={input_blob: images})
# print("elapsed time", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
