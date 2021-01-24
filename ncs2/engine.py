import logging as log
import os
import sys
from ncs2.config import Config
from ncs2.imageproc import ImageProc
from ncs2.stats import Stats

class Engine:

    def __init__(self, message, version, log_level=log.INFO):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        config = Config()
        args = config.parse()
        if not os.path.exists(args.input):
            log.error(f"{args.input} not found")
            exit(4)
        args.verbose = 0
        log.info(f"{message} {version}")
        self.args = args
