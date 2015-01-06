import bquery
import os
import random
import itertools
import tempfile
import numpy as np
import shutil
from nose.tools import assert_list_equal
from nose.plugins.skip import SkipTest
import itertools as itt


class TestCtable():
    def setup(self):
        print 'TestCtable.setup'
        self.rootdir = None

    def teardown(self):
        print 'TestCtable.teardown'
        if self.rootdir:
            shutil.rmtree(self.rootdir)
            self.rootdir = None

    def gen_almost_unique_row(self, N):
        pool = itertools.cycle(['a', 'b', 'c', 'd', 'e'])
        pool_b = itertools.cycle([1.1, 1.2])
        pool_c = itertools.cycle([1, 2, 3])
        pool_d = itertools.cycle([1, 2, 3])
        for _ in range(N):
            d = (
                pool.next(),
                pool_b.next(),
                pool_c.next(),
                pool_d.next(),
                random.random(),
                random.randint(- 10, 10),
                random.randint(- 10, 10),
            )
            yield d

    def helper_itt_groupby(self, data, keyfunc):
        groups = []
        uniquekeys = []
        data = sorted(data,
                      key=keyfunc)  # mandatory before calling itertools groupby!
        for k, g in itt.groupby(data, keyfunc):
            groups.append(list(g))  # Store group iterator as a list
            uniquekeys.append(k)

        result = {
            'groups': groups,
            'uniquekeys': uniquekeys
        }
        return result

    def test_groupby_01(self):
        """
        test_groupby_01: Test groupby's group creation
                         (groupby single row rsults into multiple groups)
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f4', 'f5', 'f6']
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print result_bcolz

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print uniquekeys

        assert_list_equal(list(result_bcolz['f0']), uniquekeys)

    @SkipTest
    def test_groupby_02(self):
        """
        test_groupby_02: Test groupby's group creation
                         (groupby over multiple rows results
                         into multiple groups)
        """
        # TODO: check group order
        random.seed(1)

        groupby_cols = ['f0', 'f1', 'f2']
        groupby_lambda = lambda x: [x[0], x[1], x[2]]
        agg_list = ['f4', 'f5', 'f6']
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print result_bcolz

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print uniquekeys

        assert_list_equal(
            [list(x) for x in result_bcolz[groupby_cols]], result_itt)

    def test_groupby_03(self):
        """
        test_groupby_03: Test groupby's aggregations
                         (groupby single row rsults into multiple groups)
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f4', 'f5', 'f6']
        agg_lambda = lambda x: [x[4], x[5], x[6]]
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print result_bcolz

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print uniquekeys

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += row[4]
                f5 += row[5]
                f6 += row[6]
            ref.append([f0, f4, f5, f6])

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)
