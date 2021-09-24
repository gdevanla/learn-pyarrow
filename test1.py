import datetime
import functools
import multiprocessing as mp
import sys
import time
from pprint import pprint
from timeit import Timer

import numpy as np
import psutil
import pyarrow as pa


class Timer:
    """A simple object for timing."""

    def __init__(self):
        """Initializing always starts the timer."""
        self.start()

    def start(self):
        """Explicit start method; will clear previous values. Start always happens on initialization."""

        self._start = time.time()
        self._stop = None
        self._past_stops = [self._start]

    def stop(self):
        self._stop = time.time()
        self._past_stops.append(self._stop)

    def clear(self):
        self._stop = None
        self._start = None

    def __call__(self):
        stop = self._stop if self._stop is not None else time.time()
        self._past_stops.append(stop)
        return stop - self._start

    def __str__(self):
        """Reports current time or, if stopped, stopped time."""
        duration = self.__call__()
        return str(datetime.timedelta(seconds=duration))
        # return str(round(duration, 4))

    def delta(self):
        """Return delta from previous calls or __str__ calls. Does not add an additional stop."""
        if len(self._past_stops) > 1:
            return self._past_stops[-1] - self._past_stops[-2]


def get_numpy_array(rows, cols):
    data = np.array(list(np.array(list(range(0, rows))) for i in range(cols)))
    return data


def run_timer(f):
    """Decorator that times and reports time of a function call"""

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        print(
            "run_timer: <start>",
            f.__name__,
        )
        t = Timer()

        post = f(*args, **kwargs)

        print(
            "run_timer:  <stop>",
            f.__name__,
            str(t),
        )

        return post

    return wrapped


def get_batch(rows, cols):
    data = get_numpy_array(rows, cols)
    # batch = pa.record_batch(data, names=[f"{x}" for x in range(0, cols)])
    batch = pa.RecordBatch.from_arrays(
        [pa.array(data[i]) for i in range(cols)],
        names=[f"{x}" for x in range(0, cols)],
    )
    return batch


def get_sink(batch):
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    for i in range(5):
        writer.write_batch(batch)
        writer.close()
    buf = sink.getvalue()
    return buf


def process_buffer(args):
    buf, index = args
    reader = pa.ipc.open_stream(buf)
    batches = [b for b in reader]
    values = batches[0][index]


def run_with_buffer(buf, rows, cols):
    with mp.Pool(5) as p:
        p.map(process_buffer, [(buf, i) for i in range(cols)])


# numpy related funcs
def process_numpy(args):
    data, index = args
    x = np.sum(data[index])
    # print(psutil.Process(os.getpid()))


def run_with_numpy(data, row, cols):
    with mp.Pool(5) as p:
        p.map(process_numpy, [(data, i) for i in range(cols)])


if __name__ == "__main__":
    for r, c in ((100, 100), (10000, 100), (100000, 100), (1000000, 100)):
        rows = r
        cols = c

        # data = get_batch(rows, cols)
        # buf = get_sink(data)
        # print(r, c, data.nbytes)
        # run_timer(run_with_buffer)(buf, rows, cols)

        data = get_numpy_array(rows, cols)
        print(r, c, data.nbytes)
        run_timer(run_with_numpy)(data, rows, cols)
