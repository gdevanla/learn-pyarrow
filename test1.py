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
import pyarrow.compute as pc

from util import run_timer


# Data Set up routines
def get_numpy_array(rows, cols):
    data = np.array(list(np.array(list(range(0, rows))) for i in range(cols)))
    return data


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
    writer.write_batch(batch)
    writer.close()
    buf = sink.getvalue()
    return buf


# Different tests
def run_with_buffer(buf, rows, cols):
    def process_buffer(args):
        buf, index = args
        reader = pa.ipc.open_stream(buf)
        batches = [b for b in reader]
        values = batches[0][index]

    with mp.Pool(5) as p:
        p.map(process_buffer, [(buf, i) for i in range(cols)])


def process_batch(args):
    batches, index = args
    values = batches[0][index]
    pc.sum(values)


def run_with_batch(data, row, cols):
    with mp.Pool(5) as p:
        p.map(process_batch, [(data, i) for i in range(cols)])


# numpy related funcs
def process_numpy(args):
    data, index = args
    x = np.sum(data[index])
    # print(psutil.Process(os.getpid()))


def run_with_numpy(data, row, cols):
    with mp.Pool(5) as p:
        p.map(process_numpy, [(data, i) for i in range(cols)])


def capture_times_func(label, log_fp):
    def f(delta):
        with open(log_fp, "a") as f:
            result = ",".join((label, str(delta)))
            f.write(result)
            f.write("\n")

    return f


def run_test1():
    header = ",".join(
        (
            "label",
            "duration",
        )
    )
    with open("/tmp/test1.txt", "w") as f:
        f.write(f"{header}\n")

    for r, c in ((100, 100), (10000, 100), (100000, 100), (1000000, 100)):
        rows = r
        cols = c

        batch = get_batch(rows, cols)
        # print(r, c, batch.nbytes)
        callback = capture_times_func(f"batch_{rows}_{cols}", "/tmp/test1.txt")
        run_timer(run_with_batch, callback)(batch, rows, cols)

        print("run_with_numpy")  #
        data = get_numpy_array(rows, cols)  #
        # print(r, c, data.nbytes)  #
        callback = capture_times_func(f"numpy_{rows}_{cols}", "/tmp/test1.txt")
        run_timer(run_with_numpy, callback)(data, rows, cols)  #

        # data = get_batch(rows, cols)  #
        # buf = get_sink(data)  #
        # del data  #
        # print("run_with_buffer")  #
        # print(r, c, buf.size)  #
        # run_timer(run_with_buffer)(buf, rows, cols)  #
        #


if __name__ == "__main__":
    run_test1()
