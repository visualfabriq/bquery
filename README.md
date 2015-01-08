bquery
======

A query and aggregation framework for Bcolz

Building
--------
To be able to build, the package bcolz with ```carray_ext.pxd``` at least version 0.7.4-dev is needed.

```
git clone https://github.com/esc/bcolz.git
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

Performance (vbench)
--------------------
Run vbench suite
```python bquery/benchmarks/vb_suite/run_suite.py```
