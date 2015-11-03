from __future__ import print_function
# bench related imports
import numpy as np
import shutil
import bquery
import pandas as pd
import itertools as itt
import cytoolz
import cytoolz.dicttoolz
from toolz import valmap, compose
from cytoolz.curried import pluck
import blaze as blz
# other imports
import contextlib
import os
import time

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

t_elapsed = 0.0


@contextlib.contextmanager
def ctime(message=None):
    "Counts the time spent in some context"
    global t_elapsed
    t_elapsed = 0.0
    print('\n')
    t = time.time()
    yield
    if message:
        print(message + ":  ", end='')
    t_elapsed = time.time() - t
    print(round(t_elapsed, 4), "sec")


ga = itt.cycle(['ES', 'NL'])
gb = itt.cycle(['b1', 'b2', 'b3', 'b4', 'b5'])
gx = itt.cycle([1, 2])
gy = itt.cycle([-1, -2])
rootdir = 'bench-data.bcolz'
if os.path.exists(rootdir):
    shutil.rmtree(rootdir)

n_rows = 1000000
print('Rows: ', n_rows)

# -- data
z = np.fromiter(((a, b, x, y) for a, b, x, y in izip(ga, gb, gx, gy)),
                dtype='S2,S2,i8,i8', count=n_rows)

ct = bquery.ctable(z, rootdir=rootdir, )
print(ct)

# -- pandas --
df = pd.DataFrame(z)
with ctime(message='pandas'):
    result = df.groupby(['f0'])['f2'].sum()
print(result)
t_pandas = t_elapsed

# -- cytoolz --
with ctime(message='cytoolz over bcolz'):
    # In Memory Split-Apply-Combine
    # http://toolz.readthedocs.org/en/latest/streaming-analytics.html?highlight=reduce#split-apply-combine-with-groupby-and-reduceby
    r = cytoolz.groupby(lambda row: row.f0, ct)
    result = valmap(compose(sum, pluck(2)), r)
print('x{0} slower than pandas'.format(round(t_elapsed / t_pandas, 2)))
print(result)

# -- blaze + bcolz --
blaze_data = blz.Data(ct.rootdir)
expr = blz.by(blaze_data.f0, sum_f2=blaze_data.f2.sum())
with ctime(message='blaze over bcolz'):
    result = blz.compute(expr)
print('x{0} slower than pandas'.format(round(t_elapsed / t_pandas, 2)))
print(result)

# -- bquery --
with ctime(message='bquery over bcolz'):
    result = ct.groupby(['f0'], ['f2'])
print('x{0} slower than pandas'.format(round(t_elapsed / t_pandas, 2)))
print(result)

ct.cache_factor(['f0'], refresh=True)
with ctime(message='bquery over bcolz (factorization cached)'):
    result = ct.groupby(['f0'], ['f2'])
print('x{0} slower than pandas'.format(round(t_elapsed / t_pandas, 2)))
print(result)

shutil.rmtree(rootdir)
