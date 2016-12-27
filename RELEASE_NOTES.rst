========================
Release notes for bquery
========================

Changes from 0.2.3 to 0.2.4
=======================

- Aggregation list can be empty now, delivering just the groupby dimensions


Changes from 0.2.2 to 0.2.3
=======================

- Optimized cython to prevent incorrect emits and to simplify logic by some duplication instead of if/then/else inside large loops
- Improved multi-processing collision avoidance during the generation of factor columns


Changes from 0.2.1 to 0.2.2
=======================

- Pip package setup (no more pre-setup dependency on numpy)


Changes from 0.2.0 to 0.2.1
=======================

- Improved dependency handling during setup


Changes from 0.1.0 to 0.2.0
=======================

- Greatly reduced memory usage
- Groupby factorization caching (will greatly enhance group by queries after the first call)
- Improved filter and groupby performance (by move towards ctable iter with tuples)
- Removed groupby maximum number of columns bottleneck (by using the cpython tuple hash calculation
- Improved count distinct performance
- Solved issues with pip v8 and numpy version dependencies
- Added "auto_cache" attribute that will cache factorizations (numerical representations of the input) automatically


Release  0.1.0
=======================
- Inital release

.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End: