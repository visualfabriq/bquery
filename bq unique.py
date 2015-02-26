import bquery as bq
import bcolz as bz
import shutil
import os


N = 1000
rootdir = 'bcolz-tmp'
if os.path.exists(rootdir):
    shutil.rmtree(rootdir)
ct = bz.fromiter(((i % 30, i % 40) for i in xrange(N)), dtype='i4,f8', count=N,
                 rootdir=rootdir)

ct = bq.open(rootdir)
print ct.unique(['f0', 'f1'])
