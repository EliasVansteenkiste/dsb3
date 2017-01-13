import time

class Timer(float):
    def __init__(self):
        self.total_time = 0
        self.start_time = None

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("Timer was already started")
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer was never started")
        self.total_time += time.time()-self.start_time
        self.start_time = None

    def reset(self):
        self.total_time = 0
        self.start_time = None

    def __float__(self):
        if self.start_time is None:
            return self.total_time
        else:
            return self.total_time + (time.time()-self.start_time)

    def __str__(self):
        res = float(self)
        if res<1:
            return "%dms" % (1000*float(self))
        elif res<10:
            return "%.2fs" % float(self)
        elif res<100:
            return "%.1fs" % float(self)
        else:
            return "%ds" % float(self)