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

import codecs
import os

from setuptools import setup, Extension, find_packages
from os.path import abspath
from sys import version_info as v
from Cython.Build import cythonize, build_ext
import numpy


# Check this Python version is supported
if v < (3, 11):
    raise Exception("Unsupported Python version %d.%d. Requires Python >= 3.11" % v[:2])


HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def get_version():
    with codecs.open(abspath('VERSION'), "r", "utf-8") as f:
        return f.readline().rstrip('\n')


# Sources & libraries
sources = ['bquery/ctable_ext.pyx']
optional_libs = ['numexpr>=1.4.1']
install_requires = [
    'pip>=8.1.2',
    'setuptools>=27.3',
    'cython>=0.29.2',
    'bcolz>=1.2.1',
    'numpy>=2'
]
setup_requires = ['numpy']
tests_requires = ['pytest', 'nose']

extras_requires = [
    'numexpr>=1.4.1'
]

extensions = [
    Extension("bquery.ctable_ext", ["bquery/ctable_ext.pyx"],
              include_dirs=[abspath('bquery'), numpy.get_include()],
              libraries=[],
              library_dirs=[])
]
ext_modules = cythonize(extensions)

package_data = {'bquery': ['ctable_ext.pxd']}
classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
    ]

setup(
    name="bquery",
    version=get_version(),
    description='A query and aggregation framework for Bcolz',
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    author='Carst Vaartjes',
    author_email='cvaartjes@visualfabriq.com',
    maintainer='Carst Vaartjes',
    maintainer_email='cvaartjes@visualfabriq.com',
    url='https://github.com/visualfabriq/b4query',
    license='MIT',
    platforms=['any'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_requires,
    extras_require=dict(
        optional=extras_requires,
        test=tests_requires
    ),
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    zip_safe=True
)
