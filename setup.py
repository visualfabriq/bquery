########################################################################
#       Files based on https://github.com/Blosc/bcolz
########################################################################
#
# License: BSD
# Created: August 16, 2012
#       Author:  Francesc Alted - francesc@blosc.io
#
########################################################################

from __future__ import absolute_import

import sys
from distutils.core import Extension
from distutils.core import setup
import textwrap

import os



########### Some utils for version checking ################

# Some functions for showing errors and warnings.
def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(
        initial_indent='   ', subsequent_indent='   ')

    print( ".. %s:: %s" % (kind.upper(), head))
    for line in tw.wrap(body):
        print(line)


def exit_with_error(head, body=''):
    _print_admonition('error', head, body)
    sys.exit(1)


def print_warning(head, body=''):
    _print_admonition('warning', head, body)


def check_import(pkgname, pkgver):
    try:
        mod = __import__(pkgname)
    except ImportError:
        exit_with_error(
            "You need %(pkgname)s %(pkgver)s or greater to run bquery!"
            % {'pkgname': pkgname, 'pkgver': pkgver})
    else:
        if mod.__version__ < pkgver:
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run bquery!"
                % {'pkgname': pkgname, 'pkgver': pkgver})

    print("* Found %(pkgname)s %(pkgver)s package installed."
          % {'pkgname': pkgname, 'pkgver': mod.__version__})
    globals()[pkgname] = mod


########### Check versions ##########

# The minimum version of Cython required for generating extensions
min_cython_version = '0.20'
# The minimum version of bcolz required
min_bcolz_version = '0.7.3.dev'


# Check if Cython is installed or not (requisite)
try:
    import Cython
    cur_cython_version = Cython.__version__
    from Cython.Distutils import build_ext
except:
    exit_with_error(
        "You need %(pkgname)s %(pkgver)s or greater to compile bcolz!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version})

if cur_cython_version < min_cython_version:
    exit_with_error(
        "At least Cython %s is needed so as to generate extensions!"
        % (min_cython_version))
else:
    print("* Found %(pkgname)s %(pkgver)s package installed."
          % {'pkgname': 'Cython', 'pkgver': cur_cython_version})

#  Check for bcolz
# check_import('bcolz', min_bcolz_version)


########### End of checks ##########


########### Project specific command line options ###########

class bquery_build_ext(build_ext):
    user_options = build_ext.user_options + \
        [
        ('from-templates', None,
         "rebuild project from code generation templates"),
        ]

    def initialize_options(self):
        self.from_templates = False
        build_ext.initialize_options(self)

    def run(self):
        if self.from_templates:
            try:
                import jinja2
            except:
                exit_with_error(
                    "You need the python package jinja2 to rebuild the " + \
                    "extension from the templates")
            execfile("bquery/templates/run_templates.py")

        build_ext.run(self)


######### End project specific command line options #########


# bquery version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('bquery/version.py', 'w').write('__version__ = "%s"\n' % VERSION)


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
# Allow setting the Blosc dir if installed in the system
BLOSC_DIR = os.environ.get('BLOSC_DIR', '')

# Sources & libraries
inc_dirs = ['bquery']
lib_dirs = []
libs = []
def_macros = []
sources_ctable = ["bquery/ctable_ext.pyx"]

# Include NumPy header dirs
from numpy.distutils.misc_util import get_numpy_include_dirs

inc_dirs.extend(get_numpy_include_dirs())
optional_libs = []

classifiers = """\
"""
setup(name="bquery",
      version=VERSION,
      description='A query and aggregation framework for Bcolz',
      long_description="""\

      """,
      classifiers=filter(None, classifiers.split("\n")),
      url='https://github.com/visualfabriq/bquery',
      license='http://www.opensource.org/licenses/bsd-license.php',
      platforms=['any'],
      cmdclass={'build_ext': bquery_build_ext},
      ext_modules=[
          Extension("bquery.ctable_ext",
                    include_dirs=inc_dirs,
                    define_macros=def_macros,
                    sources=sources_ctable,
                    library_dirs=lib_dirs,
                    libraries=libs,
                    extra_link_args=LFLAGS,
                    extra_compile_args=CFLAGS),
      ],
      packages=['bquery', 'bquery.tests'],
)
