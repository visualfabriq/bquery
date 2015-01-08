bquery
======

A query and aggregation framework for Bcolz

Building
--------
To be able to build, the package bcolz with ```carray_ext.pxd``` is needed.

```
git clone https://github.com/esc/bcolz.git
cd bcolz
git checkout pxd_v3
python setup.py build_ext --inplace
cd ..
export PYTHONPATH=$(pwd)/bcolz:${PYTHONPATH}

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
