bquery
======

A query and aggregation framework for Bcolz

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


cytoolz over bcolz:  1.8483 sec
x28.6969 slower than pandas
{'NL': 1000000, 'ES': 500000}


bquery over bcolz:  0.1969 sec
x3.0573 slower than pandas
[('ES', 500000) ('NL', 1000000)]


bquery over bcolz (factorization cached):  0.1283 sec
x1.992 slower than pandas
[('ES', 500000) ('NL', 1000000)]
```
For details about these results see please the python script

Performance (vbench)
--------------------
Run vbench suite  
```python bquery/benchmarks/vb_suite/run_suite.py```
