from contextlib import contextmanager
import tempfile
import os
import random
import shutil
import time

import bcolz as bz

import bquery as bq


@contextmanager
def ctime(message=None):
    "Counts the time spent in some context"
    t = time.time()
    yield
    if message:
        print message + ":\t",
    print round(time.time() - t, 4), "sec"


@contextmanager
def on_disk_data_cleaner(generator):
    rootdir = tempfile.mkdtemp(prefix='bcolz-')
    os.rmdir(rootdir)  # folder should be emtpy
    ct = bz.fromiter(generator, dtype='i4,i4', count=N, rootdir=rootdir)
    ct = bq.open(rootdir)
    # print ct
    ct.flush()
    ct = bq.open(rootdir)

    yield ct

    shutil.rmtree(rootdir)


def gen(N):
    x = 0
    for i in range(N):
        if random.randint(0, 1):
            x += 1
        yield x, random.randint(0, 20)


if __name__ == '__main__':
    N = int(1e5)
    g = gen(N)

    with on_disk_data_cleaner(g) as ct:
        f1 = ct['f1']
        barr = bz.eval("f1 == 1")  # filter
        with ctime('is_in_ordered_subgroups'):
            result = ct.is_in_ordered_subgroups(basket_col='f0', bool_arr=barr)
