from datetime import datetime

from vbench.api import Benchmark

common_setup = """
import bquery
import numpy as np

import random
import os
import tempfile
import itertools as itt
"""

setup = common_setup + """
def gen_almost_unique_row(N):
    pool = itt.cycle(['a', 'b', 'c', 'd', 'e'])
    pool_b = itt.cycle([1.1, 1.2])
    pool_c = itt.cycle([1, 2, 3])
    pool_d = itt.cycle([1, 2, 3])
    for _ in range(N):
        d = (
            pool.next(),
            pool_b.next(),
            pool_c.next(),
            pool_d.next(),
            random.random(),
            random.randint(- 10, 10),
            random.randint(- 10, 10),
        )
        yield d

random.seed(1)

groupby_cols = ['f0']
groupby_lambda = lambda x: x[0]
agg_list = ['f4', 'f5', 'f6']
num_rows = 100000

# -- Data --
g = gen_almost_unique_row(num_rows)
data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

rootdir = tempfile.mkdtemp(prefix='bcolz-')
os.rmdir(rootdir)  # folder should be emtpy
fact_bcolz = bquery.ctable(data, rootdir=rootdir)
fact_bcolz.flush()
fact_bcolz.cache_factor(groupby_cols, refresh=True)
result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
"""

stmt2 = "time.sleep(1)"
bm_groupby2 = Benchmark(stmt2, setup, name="GroupBy test 1",
                        start_date=datetime(2011, 7, 1))
