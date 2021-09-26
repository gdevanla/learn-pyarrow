import datetime
import functools
import multiprocessing as mp
import os
import sys
import time
from pprint import pprint
from timeit import Timer

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import ray

from util import run_timer


def get_sample_data_arrow(rows):
    get_bool_array = lambda: pa.array(
        (True if i % 2 == 0 else None for i in range(rows))
    )

    x = [pa.array(np.random.rand(rows)) for k in range(200)]
    data = [
        pa.array([f"sec_{i}" for i in range(rows)]),
        pa.array(np.random.rand(rows)),
        *x,
    ]
    batch = pa.RecordBatch.from_arrays(data, names=[f"{x}" for x in range(len(data))])
    return batch


def process_batch(args):
    batches, index = args
    values = batches[index]
    pc.sum(values)
    return psutil.Process(os.getpid()).memory_info().rss


def run_with_batch(data, row, cols):
    with mp.Pool(5) as p:
        return sum(p.map(process_batch, [(data, i) for i in range(1, cols)])) / cols


def process_batch_arrow_file(args):
    index = args[0]
    with pa.OSFile("/tmp/sample.arrow", "rb") as source:
        batches = pa.ipc.open_file(source).read_all()
        values = pc.sum(batches[index])
        return psutil.Process(os.getpid()).memory_info().rss


def run_with_batch_arrow_file(batch, cols):
    with pa.OSFile("/tmp/sample.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, batch.schema) as writer:
            writer.write_table(batch)

    with mp.Pool(5) as p:
        return (
            sum(p.map(process_batch_arrow_file, [(i,) for i in range(1, cols)])) / cols
        )


def process_batch_mapped(args):
    index = args[0]

    with pa.memory_map("/tmp/sample.arrow", "r") as source:
        batches = pa.ipc.RecordBatchFileReader(source).read_all()
        values = pc.sum(batches[index])
        return psutil.Process(os.getpid()).memory_info().rss


def run_with_batch_mapped(batch, cols):
    with pa.OSFile("/tmp/sample.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, batch.schema) as writer:
            writer.write_table(batch)

    with mp.Pool(5) as p:
        return sum(p.map(process_batch_mapped, [(i,) for i in range(1, cols)])) / cols


def capture_times_func(label, log_fp):
    def f(delta):
        with open(log_fp, "a") as f:
            result = ",".join((label, str(delta)))
            f.write(result)
            f.write("\n")

    return f


def run_test4():
    filename = "/tmp/test4.txt"
    header = ",".join(
        (
            "label",
            "duration",
        )
    )
    with open(filename, "w") as f:
        f.write(f"{header}\n")

    memory_used = dict()
    for rows in (
        10_000,
        100_000,
        500_000,
        1_000_000,
        # 2_000_000,
    ):  # , (10000, 100), (100000, 100), (1000000, 100)):

        print(f"Running for {rows=}")
        # df = get_sample_data(rows)
        table = pa.Table.from_batches([get_sample_data_arrow(rows)])

        callback = capture_times_func(f"batch_{rows}_{200}", filename)
        result = run_timer(run_with_batch, callback)(table, rows, 200)
        memory_used[f"batch_{rows}_{200}"] = result

        callback = capture_times_func(f"arrow-file_{rows}_{200}", filename)
        result = run_timer(run_with_batch_arrow_file, callback)(
            table, table.num_columns
        )
        memory_used[f"arrow-file_{rows}_{200}"] = result

        callback = capture_times_func(f"mapped-arrow-file_{rows}_{200}", filename)  #
        result = run_timer(run_with_batch_mapped, callback)(table, table.num_columns)
        memory_used[f"mapped-arrow-file_{rows}_{200}"] = result

        del table
        # run_timer(run_with_batch_map_shared)(table, table.num_columns)

    with open("/tmp/memory_test4.txt", "w") as f:
        f.write(f"label,size\n")
        for label, size in memory_used.items():
            f.write(f"{label},{size}\n")


if __name__ == "__main__":
    run_test4()
