import os
from argparse import ArgumentParser, SUPPRESS

def build_argparser():
    parser = ArgumentParser(add_help=False)
    opt = parser.add_argument_group('Options')

    opt.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      default=os.environ['HOME'] + "/data/imagen",
                      type=str)
    opt.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.",
                      default="models/ir/public/squeezenet1.1/FP16/squeezenet1.1.xml",
                      type=str)
    opt.add_argument("--labels", help="Optional. Path to a labels mapping file",
                      default="models/ir/public/squeezenet1.1/FP16/squeezenet1.1.labels",
                      type=str)

    opt.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    opt.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: "
                           "is acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    opt.add_argument("-s", "--start", help="Optional. Start index (when directory)", default=0, type=int)
    opt.add_argument("-n", "--number", help="Optional. Max number of images to process", default=10, type=int)
    opt.add_argument("-q", "--quiet", help="Optional. Quiet mode - don't write to the output", default=False,
                      type=bool)
    opt.add_argument("-tn", "--top", help="Optional. Number of top results", default=3, type=int)
    opt.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args = parser.parse_args()
    if args.number > 100:
        args.number = 100
    return args

# todo
# @timeit
# def do_process(**kwargs):
#         name = kw.get('log_name', method.__name__.upper())
#    return exec_net.infer(inputs={input_blob: images})
    # print("elapsed time", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
