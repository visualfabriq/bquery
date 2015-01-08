from __future__ import print_function
# bench related imports
import numpy as np
import bquery
import pandas as pd
import itertools as itt
import cytoolz
import cytoolz.dicttoolz
from toolz import valmap, compose
from cytoolz.curried import pluck
# other imports
import contextlib
import tempfile
import os
import time


@contextlib.contextmanager
def ctime(message=None):
    "Counts the time spent in some context"
    print('\n')
    t = time.time()
    yield
    if message:
        print(message + ":  ", end='')
    print(round(time.time() - t, 4), "sec")


ga = itt.cycle(['ES', 'NL'])
gb = itt.cycle(['b1', 'b2', 'b3', 'b4', 'b5'])
gx = itt.cycle([1, 2])
gy = itt.cycle([-1, -2])
rootdir = tempfile.mkdtemp(prefix='bcolz-')
os.rmdir(rootdir)  # tests needs this cleared

n_rows = 1000000
print('Rows: ', n_rows)
z = np.fromiter(((a, b, x, y) for a, b, x, y in itt.izip(ga, gb, gx, gy)),
                dtype='S2,S2,i8,i8', count=n_rows)

ct = bquery.ctable(z, rootdir=rootdir, )

# -- cytoolz --
with ctime(message='cytoolz'):
    # In Memory Split-Apply-Combine
    # http://toolz.readthedocs.org/en/latest/streaming-analytics.html?highlight=reduce#split-apply-combine-with-groupby-and-reduceby
    r = cytoolz.groupby(lambda row: row.f0, ct)
    result = valmap(compose(sum, pluck(2)), r)
print(result)

# -- bcolz --
with ctime(message='bcolz'):
    ct.cache_factor(['f0'], refresh=True)
    result = ct.groupby(['f0'], ['f2'])
print(result)

# -- pandas --
with ctime(message='pandas'):
    df = pd.DataFrame(z)
    result = df.groupby(['f0'])['f2'].sum()
print(result)
