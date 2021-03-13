import sys
import time

"""
to execute from top dir
cd $HOME/github/myriad-playground
python3 -m insg.oak.main
"""
from insg.common import Stats

if __name__ == "__main__":
    print("Hello from main.py")
    # print(sys.path)
    s = Stats()
    s.begin()
    s.mark()
    time.sleep(0.03)
    s.bump()
    s.mark()
    time.sleep(0.04)
    s.bump(is_error=True)
    s.end()
    print(s)


