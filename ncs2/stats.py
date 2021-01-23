import sys
import time
import logging as log


_pv = sys.version_info
if _pv[0] == 3 and _pv[1] == 6:
    # if using py 3.6 - backport of 3.7 dataclasses
    from dataclasses import dataclass

@dataclass
class Stats:
    begin_time: float
    end_time: float
    process_duration: float
    mark_time: float
    total_count: int
    failed: int

    def __init__(self):
        self.process_duration = 0
        self.total_count = 0
        self.failed_count = 0
        self.begin()
        self.end()

    def begin(self):
        self.begin_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()

    def mark(self):
        self.mark_time = time.perf_counter()

    def bump(self, is_error=False):
        delta = time.perf_counter() - self.mark_time
        if delta > 0:
            self.process_duration += delta
        self.total_count += 1
        if is_error:
            self.failed_count += 1

    def __str__(self):
        if self.begin_time > self.end_time:
            self.end_time = time.perf_counter()
        dur = self.end_time - self.begin_time
        str = f"  Elapsed time: {dur*1000:.0f} ms\n"
        if dur > 0 and self.total_count > 0:
            avg = (self.process_duration * 1000) / self.total_count
            str += f"\n  Total images: {self.total_count}, failed: {self.failed_count}\n"
            str += f"Inference time: {self.process_duration*1000:.0f} ms\n"
            str += f"       Average: {avg:.2f} ms\n"
        return str


if __name__ == '__main__':
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
