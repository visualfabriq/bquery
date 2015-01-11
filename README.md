bquery
======

A query and aggregation framework for Bcolz. 

Bcolz is a light weight package that provides columnar, chunked data containers that can be compressed either in-memory and on-disk. that are compressed by default not only for reducing memory/disk storage, but also to improve I/O speed. It excels at storing and sequentially accessing large, numerical data sets.

The bquery framework provides methods to perform query and aggregation operations on bcolz containers, as well as accelerate these operations by pre-processing possible groupby columns. Currently the real-life performance of sum aggregations using <i>on-disk bcolz</i> queries is normally between 1.5 and 3.0 times slower than similar <i>in-memory Pandas</i> aggregations. See the Benchmark paragraph below.

It is important to notice that while the end result is a bcolz ctable (which can be out-of-core) and the input can be any out-of-core ctable, the intermediate result will be an in-memory numpy array. This is because most groupby operations on non-sorted tables require random memory access while bcolz is limited to sequential access for optimum performance. However, this memory footprint is limited to the groupby result length and can be further optimized in the future to a per-column usage.

At the moment, only two aggregation methods are provided: sum and sum_na (which ignores nan values), but we aim to extend this to all normal operations in the future.
Other planned improvements are further improving per-column parallel execution of a query and extending numexpr with in/not in functionality to further speed up advanced filtering.

Though nascent, the technology itself is reliable and stable, if still limited in the depth of functionality. Visualfabriq uses bcolz and bquery to reliably handle billions of records for our clients with real-time reporting and machine learning usage. 

Bquery requires bcolz. The use is also greatly encouraged to install numexpr.

Building
--------
To be able to build, the package bcolz with ```carray_ext.pxd``` at least version 0.8.0 is needed.

```
git clone https://github.com/blosc/bcolz.git
cd bcolz
python setup.py build_ext --inplace
cd ..
export PYTHONPATH=$(pwd)/bcolz:${PYTHONPATH}
```

```
python setup.py build_ext --inplace
```

Installing
----------
```
python setup.py install
```

Testing
-------
```nosetests bquery```

Benchmarks
----------
Short benchmark to compare bquery, cytoolz & pandas  
```python bquery/benchmarks/bench_groupby.py```

Results might vary depending on where testing is performed  

Note: ctable is in this case on-disk storage vs pandas in-memory  

```
Groupby on column 'f0'
Aggregation results on column 'f2'
Rows:  1000000

ctable((1000000,), [('f0', 'S2'), ('f1', 'S2'), ('f2', '<i8'), ('f3', '<i8')])
  nbytes: 19.07 MB; cbytes: 1.14 MB; ratio: 16.70
  cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  rootdir := '/var/folders/_y/zgh0g75d13d65nd9_d7x8llr0000gn/T/bcolz-LaL2Hn'
[('ES', 'b1', 1, -1) ('NL', 'b2', 2, -2) ('ES', 'b3', 1, -1) ...,
 ('NL', 'b3', 2, -2) ('ES', 'b4', 1, -1) ('NL', 'b5', 2, -2)]


pandas:  0.0703 sec
f0
ES     500000
NL    1000000
Name: f2, dtype: int64


cytoolz over bcolz:  1.7976 sec
x25.5802 slower than pandas
{'NL': 1000000, 'ES': 500000}


bquery over bcolz:  0.1876 sec
x2.6697 slower than pandas
[('ES', 500000) ('NL', 1000000)]


bquery over bcolz (factorization cached):  0.1426 sec
x2.0292 slower than pandas
[('ES', 500000) ('NL', 1000000)]
```
For details about these results see please the python script

Performance (vbench)
--------------------
Run vbench suite  
```python bquery/benchmarks/vb_suite/run_suite.py```
