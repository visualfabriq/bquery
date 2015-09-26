bquery
======

A query and aggregation framework for Bcolz.

Bcolz is a light weight package that provides columnar, chunked data containers that can be compressed either in-memory and on-disk. that are compressed by default not only for reducing memory/disk storage, but also to improve I/O speed. It excels at storing and sequentially accessing large, numerical data sets.

The bquery framework provides methods to perform query and aggregation operations on bcolz containers, as well as accelerate these operations by pre-processing possible groupby columns. Currently the real-life performance of sum aggregations using <i>on-disk bcolz</i> queries is normally between 1.5 and 3.0 times slower than similar <i>in-memory Pandas</i> aggregations. See the Benchmark paragraph below.

It is important to notice that while the end result is a bcolz ctable (which can be out-of-core) and the input can be any out-of-core ctable, the intermediate result will be an in-memory numpy array. This is because most groupby operations on non-sorted tables require random memory access while bcolz is limited to sequential access for optimum performance. However, this memory footprint is limited to the groupby result length and can be further optimized in the future to a per-column usage.

At the moment, only two aggregation methods are provided: sum and sum_na (which ignores nan values), but we aim to extend this to all normal operations in the future.
Other planned improvements are further improving per-column parallel execution of a query and extending numexpr with in/not in functionality to further speed up advanced filtering.

Though nascent, the technology itself is reliable and stable, if still limited in the depth of functionality. Visualfabriq uses bcolz and bquery to reliably handle billions of records for our clients with real-time reporting and machine learning usage.

Bquery requires bcolz. The user is also greatly encouraged to install numexpr.

Any help in extending, improving and speeding up bquery is very welcome.

Usage
--------

Bquery subclasses the ctable from bcolz, meaning that all original ctable functions are available while adding specific new ones. First start by having a ctable (if you do not have anything available, see the '''bench_groupby.py''' file for an example.

    import bquery
    # assuming you have an example on-table bcolz file called example.bcolz
    ct = bquery.ctable(rootdir='example.bcolz')

A groupby with aggregation is easy to perform:

ctable.groupby(list of groupby columns, agg_list)

The agg_list contains the aggregations operations, which can be:
- a straight forward sum of a list of columns with a similarly named output: ['m1', 'm2', ...]
- a list of new columns with input names & aggregations [['m1', 'sum'], ['m2', 'count'], ...]
- a list that includes the type of aggregation and the output of each column, i.e. [['m1', 'sum', 'm1_sum'], ['m1', 'count', 'm1_count'], ...]

Examples:

    # groupby column f0, perform a sum on column f2 and keep the output column with the same name
    ct.groupby(['f0'], ['f2'])

    # groupby column f0, perform a count on column f2
    ct.groupby(['f0'], [['f2', 'count']])

    # groupby column f0, with a sum on f2 (output to 'f2_sum') and a sum_na on f2 (output to 'f2_sum_na')
    ct.groupby(['f0'], [['f2', 'sum', 'f2_sum'], ['f2', 'sum_na', 'f2_sum_na']])

If recurrent aggregations are done (typical in a reporting environment), you can speed up aggregations by preparing factorizations of groupby columns:

ctable.cache_factor(list of all possible groupby columns)

    # cache factorization of column f0 to speed up future groupbys over column f0
    ct.cache_factor(['f0'])

If the table is changed, the factorization has to be re-performed. This is not triggered automatically yet.

Building & Installing
---------------------

To be able to build, the package bcolz with ```carray_ext.pxd``` at least version 0.8.0 is needed.

Clone bcolz build it and install it (at the moment both steps needed)

```
git clone https://github.com/blosc/bcolz.git
cd bcolz
python setup.py build_ext --inplace
cd ..
export PYTHONPATH=$(pwd)/bcolz:${PYTHONPATH}
```

Go back to your bquery directory and repeat build and install steps.
Note: ```bquery/templates/ctable_ext.template.pyx``` will be used as 
template to generate the source file ```bquery/ctable_ext.pyx``` used 
by the cython compiler, if you are developing new features remember to write 
those modifications in the template file, ```bquery/ctable_ext.pyx```
will be overwritten each time you run 
```python setup.py build_ext --from-templates --inplace```. 

```
python setup.py build_ext --from-templates --inplace
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


pandas:  0.0827 sec
f0
ES     500000
NL    1000000
Name: f2, dtype: int64


cytoolz over bcolz:  1.8612 sec
x22.5 slower than pandas
{'NL': 1000000, 'ES': 500000}


blaze over bcolz:  0.2983 sec
x3.61 slower than pandas
   f0   sum_f2
0  ES   500000
1  NL  1000000


bquery over bcolz:  0.1833 sec
x2.22 slower than pandas
[('ES', 500000) ('NL', 1000000)]


bquery over bcolz (factorization cached):  0.1302 sec
x1.57 slower than pandas
[('ES', 500000) ('NL', 1000000)]
```
For details about these results see please the python script

You could also have a look at http://nbviewer.ipython.org/github/visualfabriq/bquery/blob/ipynb_bench/bquery/benchmarks/bench_groupby.ipynb

Performance (vbench)
--------------------
Run vbench suite  
```python bquery/benchmarks/vb_suite/run_suite.py```
