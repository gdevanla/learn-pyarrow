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
import static_frame as sf

from util import run_timer


def get_sample_data_simple(rows):
    data = pd.DataFrame(
        {
            0: [f"sec_{i}" for i in range(rows)],
            1: np.random.rand(rows),
            2: np.array([True if i % 2 == 0 else None for i in range(rows)]),
            **{k: np.random.rand(rows) for k in range(200)},
        }
    )
    return data


def get_sample_data(rows):

    get_bool_array = lambda: np.array(
        [True if i % 2 == 0 else None for i in range(rows)]
    )

    data = pd.DataFrame(
        {
            0: [f"sec_{i}" for i in range(rows)],
            1: np.random.rand(rows),
            2: get_bool_array(),
            **{
                k: (np.random.rand(rows) if k % 2 == 0 else get_bool_array())
                for k in range(200)
            },
        }
    )
    return data


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


def process_batch(args):
    batches, index = args
    # df = batches.to_pandas()
    values = batches[index]
    return psutil.Process(os.getpid()).memory_info().rss / 1e6
    # print("batch=", psutil.Process(os.getpid()).memory_info().rss)
    # print(values)
    # print(values)
    # x = df[str(index)]
    # print(index, x[0])


def run_with_batch(batch, cols):
    with mp.Pool(5) as p:
        return sum(p.map(process_batch, [(batch, i) for i in range(cols)])) / cols


def process_p2b2p(args):
    batches, index = args
    df = batches.to_pandas()
    df[str(index)].sum()
    return psutil.Process(os.getpid()).memory_info().rss / 1e6
    # print("batch=", psutil.Process(os.getpid()).memory_info().rss)
    # print(values)
    # print(values)
    # x = df[str(index)]
    # print(index, x[0])


def run_with_p2b2p(df, rows, cols):
    batch = pa.Table.from_pandas(df, preserve_index=False).combine_chunks()
    with mp.Pool(5) as p:
        return sum(p.map(process_p2b2p, [(batch, i) for i in range(cols)])) / cols


# numpy related funcs
def process_with_pandas(args):
    data, index = args
    x = data[index]
    x.sum()
    return psutil.Process(os.getpid()).memory_info().rss / 1e6
    # print("pandas=", psutil.Process(os.getpid()).memory_info().rss)
    # print(index, x[0])


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


def run_with_static_frame(data, row, cols):
    with mp.Pool(5) as p:
        return sum(p.map(process_with_pandas, [(data, i) for i in range(cols)])) / cols


def run_test3():
    filename = "/tmp/test3.txt"
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
        50_000,
        100_000,
        # 500_000,
        # 750_000,  #
        # 1_000_000,  #
        # 2000000,  #  memory error from here on
        # 5000000  #
        # 100000000,
    ):  # , (10000, 100), (100000, 100), (1000000, 100)):

        # create dataframe and table upfront
        df = get_sample_data(rows)
        table = pa.Table.from_pandas(df, preserve_index=False).combine_chunks()
        print(rows, table.nbytes)
        callback = capture_times_func(f"batch_{rows}_3", filename)  #
        result = run_timer(run_with_batch, callback)(table, table.num_columns)
        memory_used[f"batch_{rows}_3"] = result

        print("run_with_pandas")  #
        print(rows)  #
        callback = capture_times_func(f"pandas_{rows}_3", filename)  #
        result = run_timer(run_with_pandas, callback)(df, rows, len(df.columns))  #
        memory_used[f"pandas_{rows}_3"] = result

        print("run_with_pandas to batch to pandas")  #
        print(rows)  #
        callback = capture_times_func(
            f"pandas-to-batch-to-pandas_{rows}_3", filename
        )  #
        result = run_timer(run_with_p2b2p, callback)(df, rows, len(df.columns))  #
        memory_used[f"p2b2p_{rows}_3"] = result

        print("run_with_static_frame")  #
        print(rows)  #
        callback = capture_times_func(f"staticframe_{rows}_3", filename)  #
        result = run_timer(run_with_static_frame, callback)(
            sf.Frame.from_pandas(df), rows, len(df.columns)
        )  #
        memory_used[f"static_{rows}_3"] = result

    with open("/tmp/memory_test3.txt", "w") as f:
        f.write(f"label,size\n")
        for label, size in memory_used.items():
            f.write(f"{label},{size}\n")


if __name__ == "__main__":
    run_test3()
