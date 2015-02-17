from nose.plugins.attrib import attr
import bquery
import os
import random
import itertools
import tempfile
import numpy as np
import shutil
import nose
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_list_equal
from nose.plugins.skip import SkipTest
import itertools as itt
from bquery.ctable import SUM_DEF, SUM_COUNT, SUM_COUNT_NA, SUM_SORTED_COUNT_DISTINCT

class TestCtable():
    def setup(self):
        print 'TestCtable.setup'
        self.rootdir = None

    def teardown(self):
        print 'TestCtable.teardown'
        if self.rootdir:
            shutil.rmtree(self.rootdir)
            self.rootdir = None

    def gen_dataset_count(self, N):
        pool = itertools.cycle(['a', 'a',
                                'b', 'b', 'b',
                                'c', 'c', 'c', 'c', 'c'])
        pool_b = itertools.cycle([0.0, 0.0,
                                  1.0,1.0,1.0,
                                  3.0,3.0,3.0,3.0,3.0])
        pool_c = itertools.cycle([0,0,1,1,1,3,3,3,3,3])
        pool_d = itertools.cycle([0,0,1,1,1,3,3,3,3,3])
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

    def gen_dataset_count_with_NA(self, N):
        pool = itertools.cycle(['a', 'a',
                                'b', 'b', 'b',
                                'c', 'c', 'c', 'c', 'c'])
        pool_b = itertools.cycle([0.0, 0.1,
                                  1.0,1.0,1.0,
                                  3.0,3.0,3.0,3.0,3.0])
        pool_c = itertools.cycle([0,0,1,1,1,3,3,3,3,3])
        pool_d = itertools.cycle([0,0,1,1,1,3,3,3,3,3])
        pool_e = itertools.cycle([np.nan, 0.0,
                                  np.nan,1.0,1.0,
                                  np.nan,3.0,3.0,3.0,3.0])
        for _ in range(N):
            d = (
                pool.next(),
                pool_b.next(),
                pool_c.next(),
                pool_d.next(),
                pool_e.next(),
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

    def test_groupby_02(self):
        """
        test_groupby_02: Test groupby's group creation
                         (groupby over multiple rows results
                         into multiple groups)
        """
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
            sorted([list(x) for x in result_bcolz[groupby_cols]]),
            sorted(uniquekeys))

    def test_groupby_03(self):
        """
        test_groupby_03: Test groupby's aggregations
                         (groupby single row results into multiple groups)
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

    def test_groupby_04(self):
        """
        test_groupby_04: Test groupby's aggregation
                             (groupby over multiple rows results
                             into multiple groups)
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
            ref.append(f0 + [f4, f5, f6])

        assert_list_equal(
            sorted([list(x) for x in result_bcolz]),
            sorted(ref))

    def test_groupby_05(self):
        """
        test_groupby_05: Test groupby's group creation without cache
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
            print result_bcolz

            # Itertools result
            print('--> Itertools')
            result_itt = self.helper_itt_groupby(data, groupby_lambda)
            uniquekeys = result_itt['uniquekeys']
            print uniquekeys

            ref = []
            for item in result_itt['groups']:
                f1 = 0
                for row in item:
                    f0 = row[0]
                    f1 += row[1]
                ref.append([f0]+[f1])

            assert_list_equal(
                sorted([list(x) for x in result_bcolz]),
                sorted(ref))

            yield self._assert_list_equal, list(result_bcolz['f0']), uniquekeys

    def test_groupby_06(self):
        """
        test_groupby_06: Groupby type SUM_COUNT
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f4', 'f5', 'f6']
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
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list,
                                          sum_type=SUM_COUNT)
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
                f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)

    def test_groupby_07(self):
        """
        test_groupby_07: Groupby type SUM_COUNT_NA
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f4', 'f5', 'f6']
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
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list,
                                          sum_type=SUM_COUNT_NA)
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
                print item
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
                                  np.nan, 3.0, 1.0, 3.0, 3.0])
        for _ in range(N):
            d = (
                pool.next(),
                pool_b.next(),
                pool_c.next(),
                pool_d.next(),
                pool_e.next(),
                random.randint(- 10, 10),
                random.randint(- 10, 10),
            )
            yield d

    @attr('dev')
    def test_groupby_08(self):
        """
        test_groupby_08: Groupby's type SUM_SORTED_COUNT_DISTINCT
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f4']
        num_rows = 20

        # -- Data --
        g = self.gen_dataset_count_with_NA_08(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')
        print 'data'
        print data

        # -- Bcolz --
        print('--> Bcolz')
        self.rootdir = tempfile.mkdtemp(prefix='bcolz-')
        os.rmdir(self.rootdir)  # folder should be emtpy
        fact_bcolz = bquery.ctable(data, rootdir=self.rootdir)
        fact_bcolz.flush()

        fact_bcolz.cache_factor(groupby_cols, refresh=True)
        result_bcolz = fact_bcolz.groupby(groupby_cols, agg_list,
                                          sum_type=SUM_SORTED_COUNT_DISTINCT)
        print result_bcolz
        #
        # # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print uniquekeys

        ref = []

        for n, (u, item) in enumerate(zip(uniquekeys, result_itt['groups'])):
            f4 = len(self._get_unique([x[4] for x in result_itt['groups'][n]]))
            # f5 = len(self._get_unique([x[5] for x in result_itt['groups'][0]]))
            # f6 = len(self._get_unique([x[6] for x in result_itt['groups'][0]]))
            ref.append([u, f4])

        assert_list_equal(
            [list(x) for x in result_bcolz], ref)

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


if __name__ == '__main__':
    nose.main()