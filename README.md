bquery
======

A query and aggregation framework for Bcolz

Building
--------
See ```.travis.yml``` install section to get bcolz sources to be able to run 
the following command.

```
git clone https://github.com/esc/bcolz.git
cd bcolz
git checkout pxd_v3
cd ..
export PYTHONPATH=$(pwd)/bcolz:${PYTHONPATH}

python setup.py build_ext --inplace
```
