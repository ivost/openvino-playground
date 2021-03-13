from argparse import ArgumentParser
# from configparser import ConfigParser, ExtendedInterpolation

import configparser
import collections
import sys
from pathlib import Path
import logging as log

class Config:
    def __init__(self, log_level=log.INFO):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        self.cp = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

    @staticmethod
    def existing_path(path):
        p = Path(path)
        if not p.exists():
            log.error(f"Path {path} not found")
            exit(4)
        return p

    def read(self, config_file):
        self.cp.read(config_file)

        for name in self.cp.sections():
            section = self.cp[name]
            d = {k: section[k] for k in section}
            t = collections.namedtuple(name, d)
            setattr(self, name, t(**d))
            log.debug(f"section {name}, {getattr(self, name)}")
        return


if __name__ == '__main__':
    c = Config()
    c.read("config.ini")
    inp: Path = Config.existing_path(c.network.model)
    assert inp.exists()

    # print(f"config.cp['network']['weights']: {c.cp['network']['weights']}")
    # print(f"network.model: {c.network.model}")
    # print(f"network: {c.network}")

    top = int(c.network.top)
    assert top == 3
