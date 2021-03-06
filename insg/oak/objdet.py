import sys
# import tkinter as tk
# from tkinter import filedialog

from insg.common.videoengine import VideoEngine

VERSION = "v.2021.3.18"


class Detect(VideoEngine):

    def __init__(self):
        super().__init__("Video object detection test", VERSION)
        self.configure()

    def configure(self):
        if len(sys.argv) == 1:
            pass
            # root = tk.Tk()
            # root.withdraw()
            # video_path = filedialog.askopenfilename(initialdir=video_path)
        else:
            if len(sys.argv) > 1:
                self.input = sys.argv[1]
            if len(sys.argv) > 2:
                self.blob = sys.argv[2]

    def main(self):
        self.define_pipeline()
        self.run_pipeline()


if __name__ == '__main__':
    c = Detect()
    c.main()
