import os
import random
import itertools
import tempfile
import shutil
import math
import itertools as itt
from contextlib import contextmanager

import nose
import numpy as np
import bcolz as bz
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from nose.tools import assert_list_equal

import bquery


class TestCtable(object):
    @contextmanager
    def on_disk_data_cleaner(self, data):
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        ct = bquery.ctable(data, rootdir=self.rootdir)
        # print(ct)
        ct.flush()
        ct = bquery.open(self.rootdir)

        yield ct

        shutil.rmtree(self.rootdir)
        self.rootdir = None

    def setup(self):
        print('TestCtable.setup')
        self.rootdir = None

    def teardown(self):
        print('TestCtable.teardown')
        if self.rootdir:
            shutil.rmtree(self.rootdir)
            self.rootdir = None

    def gen_dataset_count(self, N):
        pool = itertools.cycle(['a', 'a',
                                'b', 'b', 'b',
                                'c', 'c', 'c', 'c', 'c'])
        pool_b = itertools.cycle([0.0, 0.0,
                                  1.0, 1.0, 1.0,
                                  3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itertools.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itertools.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                random.random(),
                random.randint(- 10, 10),
                random.randint(- 10, 10),
            )
            yield d

    def gen_dataset_count_with_NA(self, N):
        pool = itertools.cycle(['a', 'a',
                                'b', 'b', 'b',
                                'c', 'c', 'c', 'c', 'c'])
        pool_b = itertools.cycle([0.0, 0.1,
                                  1.0, 1.0, 1.0,
                                  3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itertools.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itertools.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_e = itertools.cycle([np.nan, 0.0,
                                  np.nan, 1.0, 1.0,
                                  np.nan, 3.0, 3.0, 3.0, 3.0])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                next(pool_e),
                random.randint(- 10, 10),
                random.randint(- 10, 10),
            )
            yield d

    def gen_almost_unique_row(self, N):
        pool = itertools.cycle(['a', 'b', 'c', 'd', 'e'])
        pool_b = itertools.cycle([1.1, 1.2])
        pool_c = itertools.cycle([1, 2, 3])
        pool_d = itertools.cycle([1, 2, 3])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
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
        # no operation is specified in `agg_list`, so `sum` is used by default.
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
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        assert_list_equal(list(result_bcolz['f0']), uniquekeys)

    def test_groupby_02(self):
        """
        test_groupby_02: Test groupby's group creation
                         (groupby over multiple rows results
                         into multiple groups)
        """
        random.seed(1)

        groupby_cols = ['f0', 'f1', 'f2']
        groupby_lambda = lambda x: [x[0], x[1], x[2]]
        # no operation is specified in `agg_list`, so `sum` is used by default.
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
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        assert_list_equal(
            sorted([list(x) for x in result_bcolz[groupby_cols]]),
            sorted(uniquekeys))

    def test_groupby_03(self):
        """
        test_groupby_03: Test groupby's aggregations
                        (groupby single row results into multiple groups)
                        Groupby type 'sum'
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
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

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

    def test_groupby_04(self):
        """
        test_groupby_04: Test groupby's aggregation
                             (groupby over multiple rows results
                             into multiple groups)
                             Groupby type 'sum'
        """
        random.seed(1)

        groupby_cols = ['f0', 'f1', 'f2']
        groupby_lambda = lambda x: [x[0], x[1], x[2]]
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
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

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
            ref.append(f0 + [f4, f5, f6])

        assert_list_equal(
            sorted([list(x) for x in result_bcolz]),
            sorted(ref))

    def test_groupby_05(self):
        """
        test_groupby_05: Test groupby's group creation without cache
        Groupby type 'sum'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f1']
        num_rows = 200

        for _dtype in \
                [
                    'i8',
                    'i4',
                    'f8',
                    'S1',
                ]:

            # -- Data --
            if _dtype == 'S1':
                iterable = ((str(x % 5), x % 5) for x in range(num_rows))
            else:
                iterable = ((x % 5, x % 5) for x in range(num_rows))

            data = np.fromiter(iterable, dtype=_dtype + ',i8')

            # -- Bcolz --
            print('--> Bcolz')
            self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
            os.rmdir(self.rootdir)  # folder should be emtpy
            fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
            fact_bcolz.flush()

            result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
            print(result_bcolz)

            # Itertools result
            print('--> Itertools')
            result_itt = self.helper_itt_groupby(data, groupby_lambda)
            uniquekeys = result_itt['uniquekeys']
            print(uniquekeys)

            ref = []
            for item in result_itt['groups']:
                f1 = 0
                for row in item:
                    f0 = row[0]
                    f1 += row[1]
                ref.append([f0] + [f1])

            assert_list_equal(
                sorted([list(x) for x in result_bcolz]),
                sorted(ref))

            yield self._assert_list_equal, list(result_bcolz['f0']), uniquekeys

    def test_groupby_06(self):
        """
        test_groupby_06: Groupby type 'count'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count'], ['f5', 'count'], ['f6', 'count']]
        num_rows = 2000

        # -- Data --
        g = self.gen_dataset_count(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)

    def test_groupby_07(self):
        """
        test_groupby_07: Groupby type 'count'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count'], ['f5', 'count'], ['f6', 'count']]
        num_rows = 1000

        # -- Data --
        g = self.gen_dataset_count_with_NA(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                if row[4] == row[4]:
                    f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)

    def _get_unique(self, values):
        new_values = []
        nan_found = False

        for item in values:
            if item not in new_values:
                if item == item:
                    new_values.append(item)
                else:
                    if not nan_found:
                        new_values.append(item)
                        nan_found = True

        return new_values

    def gen_dataset_count_with_NA_08(self, N):
        pool = itertools.cycle(['a', 'a',
                                'b', 'b', 'b',
                                'c', 'c', 'c', 'c', 'c'])
        pool_b = itertools.cycle([0.0, 0.1,
                                  1.0, 1.0, 1.0,
                                  3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itertools.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itertools.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_e = itertools.cycle([np.nan, 0.0,
                                  np.nan, 0.0, 1.0,
                                  np.nan, 3.0, 1.0, 3.0, 1.0])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                next(pool_e),
                random.randint(- 500, 500),
                random.randint(- 100, 100),
            )
            yield d

    def test_groupby_08(self):
        """
        test_groupby_08: Groupby's type 'count_distinct'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count_distinct'], ['f5', 'count_distinct'], ['f6', 'count_distinct']]
        num_rows = 2000

        # -- Data --
        g = self.gen_dataset_count_with_NA_08(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')
        print('data')
        print(data)

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print(result_bcolz)
        #
        # # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []

        for n, (u, item) in enumerate(zip(uniquekeys, result_itt['groups'])):
            f4 = len(self._get_unique([x[4] for x in result_itt['groups'][n]]))
            f5 = len(self._get_unique([x[5] for x in result_itt['groups'][n]]))
            f6 = len(self._get_unique([x[6] for x in result_itt['groups'][n]]))
            ref.append([u, f4, f5, f6])

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)

    def test_groupby_08b(self):
        """
        test_groupby_08b: Groupby's type 'count_distinct' with a large number of records
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count_distinct'], ['f5', 'count_distinct'], ['f6', 'count_distinct']]
        num_rows = 200000

        # -- Data --
        g = self.gen_dataset_count_with_NA_08(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')
        print('data')
        print(data)

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print(result_bcolz)
        #
        # # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []

        for n, (u, item) in enumerate(zip(uniquekeys, result_itt['groups'])):
            f4 = len(self._get_unique([x[4] for x in result_itt['groups'][n]]))
            f5 = len(self._get_unique([x[5] for x in result_itt['groups'][n]]))
            f6 = len(self._get_unique([x[6] for x in result_itt['groups'][n]]))
            ref.append([u, f4, f5, f6])

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)

    def gen_dataset_count_with_NA_09(self, N):
        pool = (random.choice(['a', 'b', 'c']) for _ in range(N))
        pool_b = (random.choice([0.1, 0.2, 0.3]) for _ in range(N))
        pool_c = (random.choice([0, 1, 2, 3]) for _ in range(N))
        pool_d = (random.choice([0, 1, 2, 3]) for _ in range(N))

        pool_e = (math.ceil(x) for x in np.arange(0, N * 0.1, 0.1))
        pool_f = (math.ceil(x) for x in np.arange(0, N * 0.3, 0.3))
        pool_g = (math.ceil(x) for x in np.arange(0, N, 1))
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                # --
                next(pool_e),
                next(pool_f),
                next(pool_g),
            )
            yield d

    def test_groupby_09(self):
        """
        test_groupby_09: Groupby's type 'sorted_count_distinct'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'sorted_count_distinct'], ['f5', 'sorted_count_distinct'], ['f6', 'sorted_count_distinct']]
        num_rows = 2000

        # -- Data --
        g = self.gen_dataset_count_with_NA_09(num_rows)
        sort = sorted([item for item in g], key=lambda x: x[0])
        data = np.fromiter(sort, dtype='S1,f8,i8,i4,f8,i8,i4')
        print('data')
        print(data)

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list)
        print(result_bcolz)

        # # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []

        for n, (u, item) in enumerate(zip(uniquekeys, result_itt['groups'])):
            f4 = len(self._get_unique([x[4] for x in result_itt['groups'][n]]))
            f5 = len(self._get_unique([x[5] for x in result_itt['groups'][n]]))
            f6 = len(self._get_unique([x[6] for x in result_itt['groups'][n]]))
            ref.append([u, f4, f5, f6])
        print(ref)

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)

    def test_groupby_10(self):
        """
        test_groupby_10: Groupby's 'sorted_count_distinct', no column provided
        """
        random.seed(1)

        groupby_cols = []
        agg_list = [['f1', 'sorted_count_distinct'], ['f2', 'sorted_count_distinct']]
        num_rows = 10

        # -- Data --
        data = np.array(
            [(0, 1, 1),
             (1, 1, 1),
             (1, 2, 1),
             (0, 2, 1),
             (1, 2, 1),
             (2, 2, 1),
             (0, 3, 2),
             (0, 3, 2),
             (1, 4, 2)],
            dtype='i8,i8,i8')

        # -- Bcolz --
        with self.on_disk_data_cleaner(data) as ct:
            result_bcolz = ct.groupby(groupby_cols, agg_list)

        assert_list_equal([list(x) for x in result_bcolz], [[4, 2]])

    def test_groupby_11(self):
        """
        test_groupby_11: Groupby's 'sorted_count_distinct', pre-filter  &
                         no column provided

        """
        random.seed(1)

        groupby_cols = []
        agg_list = [['f1', 'sorted_count_distinct'], ['f2', 'sorted_count_distinct']]
        num_rows = 10

        # -- Data --
        data = np.array(
            [(0, 1, 1),
             (1, 1, 1),
             (1, 2, 1),
             (0, 2, 1),
             (1, 2, 1),
             (2, 2, 1),
             (0, 3, 2),
             (0, 3, 2),
             (1, 4, 2)],
            dtype='i8,i8,i8')

        with self.on_disk_data_cleaner(data) as ct:
            barr = ct.where_terms([('f0', 'in', [0])])
            result_bcolz = ct.groupby(groupby_cols, agg_list,
                                      bool_arr=barr)

        assert_list_equal([list(x) for x in result_bcolz], [[3, 2]])

    def test_groupby_12(self):
        """
        test_groupby_12: Groupby's 'sorted_count_distinct', no column provided
        """
        random.seed(1)

        groupby_cols = []
        agg_list = [['f1', 'sorted_count_distinct']]
        num_rows = 10

        # -- Data --
        data = np.array(
            [(0, 1),
             (1, 1),
             (1, 2),
             (0, 2),
             (1, 2),
             (2, 2),
             (0, 3),
             (0, 3),
             (1, 4)],
            dtype='i8,i8')

        with self.on_disk_data_cleaner(data) as ct:
            result_bcolz = ct.groupby(groupby_cols, agg_list)

        assert_list_equal([list(x) for x in result_bcolz], [[4]])

    def test_groupby_13(self):
        """
        test_groupby_13: Groupby's 'sorted_count_distinct', pre-filter
        """
        random.seed(1)

        groupby_cols = ['f0']
        agg_list = [['f1', 'sorted_count_distinct']]

        # -- Data --
        data = np.array(
            [(0, 1),
             (1, 1),
             (1, 2),
             (0, 2),
             (1, 2),
             (2, 2),
             (0, 3),
             (0, 3),
             (1, 4)],
            dtype='i8,i8')

        # -- Bcolz --
        with self.on_disk_data_cleaner(data) as ct:
            barr = ct.where_terms([('f0', 'in', [0, 1])])
            result_bcolz = ct.groupby(groupby_cols, agg_list,
                                      bool_arr=barr)

        assert_list_equal([list(x) for x in result_bcolz], [[0, 3], [1, 3]])

    def test_groupby_14(self):
        """
        test_groupby_14: Groupby type 'mean'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'mean'], ['f5', 'mean'], ['f6', 'mean']]
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
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = []
            f5 = []
            f6 = []
            for row in item:
                f0 = groupby_lambda(row)
                f4.append(row[4])
                f5.append(row[5])
                f6.append(row[6])

            ref.append([np.mean(f4), np.mean(f5), np.mean(f6)])

        # remove the first (text) element for floating point comparison
        result = [list(x[1:]) for x in result_bcolz]

        assert_allclose(result, ref, rtol=1e-10)

    def test_groupby_15(self):
        """
        test_groupby_15: Groupby type 'std'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'std'], ['f5', 'std'], ['f6', 'std']]
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
        print(result_bcolz)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = []
            f5 = []
            f6 = []
            for row in item:
                f0 = groupby_lambda(row)
                f4.append(row[4])
                f5.append(row[5])
                f6.append(row[6])

            ref.append([np.std(f4), np.std(f5), np.std(f6)])

        # remove the first (text) element for floating point comparison
        result = [list(x[1:]) for x in result_bcolz]

        assert_allclose(result, ref, rtol=1e-10)

    def _assert_list_equal(self, a, b):
        assert_list_equal(a, b)

    def test_where_terms00(self):
        """
        test_where_terms00: get terms in one column bigger than a certain value
        """

        # expected result
        ref_data = np.fromiter(((x > 10000) for x in range(20000)),
                               dtype='bool')
        ref_result = bquery.carray(ref_data)

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        # filter data
        terms_filter = [('f0', '>', 10000)]
        ct = bquery.ctable(data, rootdir=self.rootdir)
        result = ct.where_terms(terms_filter)

        # compare
        assert_array_equal(result, ref_result)

    def test_where_terms01(self):
        """
        test_where_terms01: get terms in one column less or equal than a
                            certain value
        """

        # expected result
        ref_data = np.fromiter(((x <= 10000) for x in range(20000)),
                               dtype='bool')
        ref_result = bquery.carray(ref_data)

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        # filter data
        terms_filter = [('f0', '<=', 10000)]
        ct = bquery.ctable(data, rootdir=self.rootdir)
        result = ct.where_terms(terms_filter)

        # compare
        assert_array_equal(result, ref_result)

    def test_where_terms02(self):
        """
        test_where_terms02: get mask where terms not in list
        """

        exclude = [0, 1, 2, 3, 11, 12, 13]

        # expected result
        mask = np.ones(20000, dtype=bool)
        mask[exclude] = False

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        # filter data
        terms_filter = [('f0', 'not in', exclude)]
        ct = bquery.ctable(data, rootdir=self.rootdir)
        result = ct.where_terms(terms_filter)

        assert_array_equal(result, mask)

    def test_where_terms03(self):
        """
        test_where_terms03: get mask where terms in list
        """

        include = [0, 1, 2, 3, 11, 12, 13]

        # expected result
        mask = np.zeros(20000, dtype=bool)
        mask[include] = True

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        # filter data
        terms_filter = [('f0', 'in', include)]
        ct = bquery.ctable(data, rootdir=self.rootdir)
        result = ct.where_terms(terms_filter)

        assert_array_equal(result, mask)

    def test_where_terms_04(self):
        """
        test_where_terms04: get mask where terms in list with only one item
        """

        include = [0]

        # expected result
        mask = np.zeros(20000, dtype=bool)
        mask[include] = True

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        # filter data
        terms_filter = [('f0', 'in', include)]
        ct = bquery.ctable(data, rootdir=self.rootdir)
        result = ct.where_terms(terms_filter)

        assert_array_equal(result, mask)

    def test_factorize_groupby_cols_01(self):
        """
        test_factorize_groupby_cols_01:
        """
        ref_fact_table = np.arange(20000) % 5
        ref_fact_groups = np.arange(5)

        # generate data
        iterable = ((x, x % 5) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')
        ct = bquery.ctable(data, rootdir=self.rootdir)

        # factorize - check the only factirized col. [0]
        fact_1 = ct.factorize_groupby_cols(['f1'])
        # cache should be used this time
        fact_2 = ct.factorize_groupby_cols(['f1'])

        assert_array_equal(ref_fact_table, fact_1[0][0])
        assert_array_equal(ref_fact_groups, fact_1[1][0])

        assert_array_equal(fact_1[0][0], fact_2[0][0])
        assert_array_equal(fact_1[1][0], fact_2[1][0])

    def test_pos_basket_01(self):
        """test_pos_basket_01:

             <----- data ----->
            | Basket | Product | Filter | Result |
            |--------|---------|--------|--------|
            | 1      | A       | 0      | 1      |
            | 1      | B       | 1      | 1      |
            | 1      | C       | 0      | 1      |
            | 2      | A       | 0      | 1      |
            | 2      | B       | 1      | 1      |
            | 3      | A       | 0      | 0      |
            | 4      | A       | 0      | 0      |
            | 4      | C       | 0      | 0      |
            | 5      | B       | 1      | 1      |
            | 6      | A       | 0      | 1      |
            | 6      | B       | 1      | 1      |
            | 6      | C       | 0      | 1      |
            | 7      | B       | 1      | 1      |
            | 7      | B       | 1      | 1      |
            | 7      | B       | 1      | 1      |
            | 8      | B       | 1      | 1      |
            | 9      | C       | 0      | 0      |

        """

        # -- Data --
        data = np.array(
            [(1, 0),
             (1, 1),
             (1, 2),
             (2, 0),
             (2, 1),
             (3, 0),
             (4, 0),
             (4, 2),
             (5, 1),
             (6, 0),
             (6, 1),
             (6, 2),
             (7, 1),
             (7, 1),
             (7, 1),
             (8, 1),
             (9, 2),
             ],
            dtype='i8,i8')

        # -- Bcolz --
        with self.on_disk_data_cleaner(data) as ct:
            f1 = ct['f1']
            barr = bz.eval("f1 == 1")  # filter
            result = ct.is_in_ordered_subgroups(basket_col='f0', bool_arr=barr,
                                                _max_len_subgroup=1)

        assert_list_equal(list(barr[:]),
                          [False, True, False, False, True, False, False, False,
                           True, False, True, False, True, True, True, True,
                           False])

        assert_list_equal(list(result[:]),
                          [True, True, True, True, True, False, False, False,
                           True, True, True, True, True, True, True, True,
                           False])


if __name__ == '__main__':
    nose.main()
