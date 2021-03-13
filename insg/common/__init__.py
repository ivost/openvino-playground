# from pkgutil import extend_path
# __path__ = extend_path('/Users/ivo/github/myriad-playground/insg', 'insg')

from .stats import Stats
from .config import Config
from .videoengine import VideoEngine

print("init2", __path__)
