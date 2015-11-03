from datetime import datetime
import os

from vbench.api import Benchmark

modules = ['vb_groupby', ]

by_module = {}
benchmarks = []

for modname in modules:
    ref = __import__(modname)
    by_module[modname] = [v for v in ref.__dict__.values()
                          if isinstance(v, Benchmark)]
    benchmarks.extend(by_module[modname])

for bm in benchmarks:
    assert (bm.name is not None)

import getpass

USERNAME = getpass.getuser()
HOME = os.path.expanduser('~')

try:
    import ConfigParser

    config = ConfigParser.ConfigParser()
    config.readfp(open(os.path.expanduser('~/.vbenchcfg')))

    REPO_PATH = config.get('setup', 'repo_path')
    REPO_URL = config.get('setup', 'repo_url')
    DB_PATH = config.get('setup', 'db_path')
    TMP_DIR = config.get('setup', 'tmp_dir')
except:
    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    REPO_URL = 'https://github.com/visualfabriq/bquery.git'
    DB_PATH = os.path.join(REPO_PATH, 'bquery/benchmarks/vb_suite/benchmarks.db')
    TMP_DIR = os.path.join(HOME, 'tmp/vb_bquery')

PREPARE = """
python setup.py clean
"""
BUILD = """
git clone https://github.com/esc/bcolz.git
cd bcolz
git checkout pxd_v3
python setup.py build_ext --inplace
cd ..
export PYTHONPATH=$(pwd)/bcolz:${PYTHONPATH}

python setup.py build_ext --inplace
"""
dependencies = []

START_DATE = datetime(2015, 1, 7)

RST_BASE = 'source'
