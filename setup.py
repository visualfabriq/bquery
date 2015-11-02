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
    raise Exception("Unsupported Python version %d.%d. Requires Python >= 2.6 "
                    "or >= 3.3." % v[:2])

import os
from os.path import abspath
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# Prevent numpy from thinking it is still in its setup process:
__builtins__.__NUMPY_SETUP__ = False


class BuildExtNumpyInc(build_ext):
    def build_extensions(self):
        from numpy.distutils.misc_util import get_numpy_include_dirs
        for e in self.extensions:
            e.include_dirs.extend(get_numpy_include_dirs())

        build_ext.build_extensions(self)


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
# Allow setting the Blosc dir if installed in the system
BLOSC_DIR = os.environ.get('BLOSC_DIR', '')

# Sources & libraries
inc_dirs = [abspath('bquery')]
lib_dirs = []
libs = []
def_macros = []
sources = ['bquery/ctable_ext.pyx']

optional_libs = []

# Handle --blosc=[PATH] --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--blosc=') == 0:
        BLOSC_DIR = os.path.expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    if arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

tests_require = []

if v < (3,):
    tests_require.extend(['unittest2', 'mock'])

# compile and link code instrumented for coverage analysis
if os.getenv('TRAVIS') and os.getenv('CI') and v[0:2] == (2, 7):
    CFLAGS.extend(["-fprofile-arcs", "-ftest-coverage"])
    LFLAGS.append("-lgcov")

setup(
    name="bquery",
    version='0.1.0.2',
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
    ],
    author='Carst Vaartjes',
    author_email='cvaartjes@visualfabriq.com',
    maintainer='Carst Vaartjes',
    maintainer_email='cvaartjes@visualfabriq.com',
    url='https://github.com/visualfabriq/bquery',
    license='BSD',
    platforms=['any'],
    ext_modules=[
        Extension(
            'bquery.ctable_ext',
            include_dirs=inc_dirs,
            define_macros=def_macros,
            sources=sources,
            library_dirs=lib_dirs,
            libraries=libs,
            extra_link_args=LFLAGS,
            extra_compile_args=CFLAGS
        )
    ],
    cmdclass={'build_ext': BuildExtNumpyInc},
    install_requires=['numpy>=1.7'],
    setup_requires=[
        'cython>=0.22',
        'numpy>=1.7',
        'setuptools>18.0',
        'setuptools-scm>1.5.4',
        # 'bcolz>=1.1.3'
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
