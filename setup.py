########################################################################
#       File based on https://github.com/Blosc/bcolz
########################################################################
#
# License: BSD
# Created: October 5, 2015
#       Author:  Carst Vaartjes - cvaartjes@visualfabriq.com
#
########################################################################


from __future__ import absolute_import

from sys import version_info as v

# Check this Python version is supported
if any([v < (2, 6), (3,) < v < (3, 3)]):
    raise Exception("Unsupported Python version %d.%d. Requires Python >= 2.7 "
                    "or >= 3.3." % v[:2])

import os
from os.path import abspath
import sys
import numpy as np

from setuptools import setup, Extension, find_packages

# Sources & libraries
inc_dirs = [abspath('bquery'), np.get_include()]
lib_dirs = []
libs = []
def_macros = []
sources = ['bquery/ctable_ext.pyx']

optional_libs = []
tests_require = []

if v < (3,):
    tests_require.extend(['unittest2', 'mock'])

setup(
    name="bquery",
    version='0.2.0.0',
    # version={
    #     'version_scheme': 'guess-next-dev',
    #     'local_scheme': 'dirty-tag',
    #     'write_to': 'bquery/version.py'
    # },
    description='A query and aggregation framework for Bcolz',
    long_description="""\

Bcolz is a light weight package that provides columnar, chunked data containers that can be compressed either in-memory and on-disk. that are compressed by default not only for reducing memory/disk storage, but also to improve I/O speed. It excels at storing and sequentially accessing large, numerical data sets.
The bquery framework provides methods to perform query and aggregation operations on bcolz containers, as well as accelerate these operations by pre-processing possible groupby columns. Currently the real-life performance of sum aggregations using on-disk bcolz queries is normally between 1.5 and 3.0 times slower than similar in-memory Pandas aggregations.

    """,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    author='Carst Vaartjes',
    author_email='cvaartjes@visualfabriq.com',
    maintainer='Carst Vaartjes',
    maintainer_email='cvaartjes@visualfabriq.com',
    url='https://github.com/visualfabriq/bquery',
    license='MIT',
    platforms=['any'],
    ext_modules=[
        Extension(
            'bquery.ctable_ext',
            include_dirs=inc_dirs,
            define_macros=def_macros,
            sources=sources,
            library_dirs=lib_dirs,
            libraries=libs
        )
    ],
    cmdclass={},
    install_requires=['numpy>=1.7', 'bcolz>=1.1.0'],
    setup_requires=[
        'cython>=0.22',
        'numpy>=1.7',
        'setuptools>18.0',
        'setuptools-scm>1.5.4',
        'bcolz>=1.1.0'
    ],
    tests_require=tests_require,
    extras_require=dict(
        optional=[
            'numexpr>=1.4.1'
        ],
        test=tests_require
    ),
    packages=find_packages(),
    package_data={'bquery': ['ctable_ext.pxd']},
    zip_safe=True
)
