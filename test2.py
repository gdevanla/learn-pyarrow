import datetime
import functools
import multiprocessing as mp
import os
import sys
import time
from pprint import pprint
from timeit import Timer

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc

from util import run_timer


def get_sample_data(rows):
    data = [
        np.array([f"sec_{i}" for i in range(rows)]),
        np.random.rand(rows),
        np.random.rand(rows),
        np.array([True if i % 2 == 0 else None for i in range(rows)]),
    ]
    return np.vstack(data)


def get_data_arrow(rows):
    data = get_sample_data(rows)
    arrow_data = []
    for column in data:
        arrow_data.append(pa.array(column))
    return arrow_data


def get_numpy_array(rows):
    return get_sample_data(rows)


def get_batch(rows):
    data = get_data_arrow(rows)
    # batch = pa.record_batch(data, names=[f"{x}" for x in range(0, cols)])
    batch = pa.RecordBatch.from_arrays(
        data,
        names=[f"{x}" for x in range(len(data))],
    )
    return batch


def get_sink(batch):
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    buf = sink.getvalue()
    return buf


def process_buffer(args):
    buf, index = args
    reader = pa.ipc.open_stream(buf)
    batches = [b for b in reader]
    values = batches[0][index]
    # print(values[0])
    # print(values)


def run_with_buffer(buf, rows, cols):
    with mp.Pool(5) as p:
        return sum(p.map(process_buffer, [(buf, i) for i in range(cols)])) / cols


def process_batch(args):
    batches, index = args
    values = batches[index]
    pc.sum(values)
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def run_with_batch(batch, rows, cols):
    with mp.Pool(5) as p:
        return (
            sum(p.map(process_batch, [(batch, i) for i in range(1, cols - 1)])) / cols
        )


# numpy related funcs
def process_numpy(args):
    data, index = args
    x = data[index]
    np.sum(x)
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def run_with_numpy(data, row, cols):
    with mp.Pool(5) as p:
        return sum(p.map(process_numpy, [(data, i) for i in range(1, cols - 1)])) / cols


# numpy related funcs
def process_with_pandas(args):
    data, index = args
    x = data[index]
    print(x[0])


def run_with_pandas(data, row, cols):
    with mp.Pool(5) as p:
        return sum(p.map(process_with_pandas, [(data, i) for i in range(cols)])) / cols


def capture_times_func(label, log_fp):
    def f(delta):
        with open(log_fp, "a") as f:
            result = ",".join((label, str(delta)))
            f.write(result)
            f.write("\n")

    return f


def run_test2():
    filename = "/tmp/test2.txt"
    header = ",".join(
        (
            "label",
            "duration",
        )
    )
    with open(filename, "w") as f:
        f.write(f"{header}\n")

    # d = get_sample_data(10)
    # a = get_data_arrow(10)
    # b = get_batch(10)

    memory_used = dict()
    for rows in (
        # 10000,
        # 100000,
        1000000,
        10000000,
        20000000,
        # 100000000,
    ):  # , (10000, 100), (100000, 100), (1000000, 100)):

        batch = get_batch(rows)
        print(rows, batch.nbytes)
        callback = capture_times_func(f"batch_{rows}_3", filename)  #
        result = run_timer(run_with_batch, callback)(batch, rows, batch.num_columns)
        memory_used[f"batch_{rows}_3"] = result

        #
        print("run_with_numpy")  #
        data = get_numpy_array(rows)
        print(rows, data.nbytes)
        callback = capture_times_func(f"numpy_{rows}_3", filename)  #
        result = run_timer(
            run_with_numpy,
            callback,
        )(data, rows, data.shape[0])
        memory_used[f"numpy_{rows}_3"] = result

        # using buffer is as good as using batch
        # data = get_batch(rows)
        # buf = get_sink(data)
        # del data  #
        # print("run_with_buffer")
        # print(rows, buf.size)  #
        # run_timer(run_with_buffer)(buf, rows, batch.num_columns)
    with open("/tmp/memory_test2.txt", "w") as f:
        f.write(f"label,size\n")
        for label, size in memory_used.items():
            f.write(f"{label},{size}\n")


if __name__ == "__main__":
    run_test2()
