import multiprocessing as mp

import pyarrow as pa


def get_batch():
    data = [
        pa.array([1, 2, 3, 4]),
        pa.array(["foo", "bar", "baz", None]),
        pa.array([True, None, False, True]),
        pa.array([5, 6, 7, 8]),
        pa.array([50, 60, 70, 80]),
        pa.array([500, 600, 700, 800]),
    ]
    batch = pa.record_batch(
        data,
        names=["f0", "f1", "f2", "f3", "f4", "f5"],
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


def process(args):
    buf, index = args
    reader = pa.ipc.open_stream(buf)
    batches = [b for b in reader]
    print(batches[0][index])


def run():
    batch = get_batch()
    buf = get_sink(batch)

    with mp.Pool(5) as p:
        p.map(process, [(buf, i) for i in range(5)])


if __name__ == "__main__":
    run()
