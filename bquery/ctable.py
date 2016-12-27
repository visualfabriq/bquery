# internal imports
from bquery import ctable_ext
import tempfile
import os
import shutil

# external imports
import numpy as np
import bcolz

class ctable(bcolz.ctable):
    def __init__(self, *args, **kwargs):
        super(ctable, self).__init__(*args, **kwargs)
        if self.rootdir and kwargs.get('auto_cache') is not False:
            self.auto_cache = True
        else:
            self.auto_cache = False

    @staticmethod
    def create_group_base_name(col_list):
        group_name = '_'.join(sorted(col_list))
        return group_name

    def cache_valid(self, col):
        """
        Checks whether the column has a factorization that exists and is not older than the source

        :param col:
        :return:
        """
        cache_valid = False

        if self.rootdir:
            col_org_file_check = self[col].rootdir + '/__attrs__'
            col_values_file_check = self[col].rootdir + '.values/__attrs__'

            if os.path.exists(col_org_file_check) and os.path.exists(col_values_file_check):
                cache_valid = os.path.getctime(col_org_file_check) < os.path.getctime(col_values_file_check)

        return cache_valid

    def group_cache_valid(self, col_list):
        """
        Checks whether the column has a factorization that exists and is not older than the source

        :param col:
        :return:
        """
        cache_valid = False

        if self.rootdir:
            col_values_file_check = os.path.join(self.rootdir, self.create_group_base_name(col_list)) + \
                                    '.values/__attrs__'

            exists_group_index = os.path.exists(col_values_file_check)
            missing_col_check = [1 for col in col_list if not os.path.exists(self[col].rootdir + '/__attrs__')]

            if exists_group_index and not missing_col_check:
                group_col_timestamp = os.path.getctime(col_values_file_check)
                col_timestamp_list = [os.path.getctime(self[col].rootdir + '/__attrs__') for col in col_list]
                cache_valid = max(col_timestamp_list) < group_col_timestamp

        return cache_valid

    def cache_factor(self, col_list, refresh=False):
        """
        Existing todos here are: these should be hidden helper carrays
        As in: not normal columns that you would normally see as a user

        The factor (label index) carray is as long as the original carray
        (and the rest of the table therefore)
        But the (unique) values carray is not as long (as long as the number
        of unique values)

        :param col_list:
        :param refresh:
        :return:
        """

        if not self.rootdir:
            raise TypeError('Only out-of-core ctables can have '
                            'factorization caching at the moment')

        if not isinstance(col_list, list):
            col_list = [col_list]

        if refresh:
            kill_list = [x for x in os.listdir(self.rootdir) if '.factor' in x or '.values' in x]
            for kill_dir in kill_list:
                shutil.rmtree(os.path.join(self.rootdir, kill_dir))

        for col in col_list:

            # create cache if needed
            if refresh or not self.cache_valid(col):
                # create directories
                col_rootdir = self[col].rootdir
                col_factor_rootdir = col_rootdir + '.factor'
                col_factor_rootdir_tmp = tempfile.mkdtemp(prefix='bcolz-')
                col_values_rootdir = col_rootdir + '.values'
                col_values_rootdir_tmp = tempfile.mkdtemp(prefix='bcolz-')

                # create factor
                carray_factor = \
                    bcolz.carray([], dtype='int64', expectedlen=self.size,
                                 rootdir=col_factor_rootdir_tmp, mode='w')
                _, values = \
                    ctable_ext.factorize(self[col], labels=carray_factor)
                carray_factor.flush()

                if not self.cache_valid(col):
                    # we check again for cache valid because in multi-processing scenarios,
                    # another process can have created and filled this factor column in the mean time
                    shutil.rmtree(col_factor_rootdir, ignore_errors=True)
                    shutil.move(col_factor_rootdir_tmp, col_factor_rootdir)

                    # create values
                    carray_values = \
                        bcolz.carray(np.fromiter(values.values(), dtype=self[col].dtype),
                                     rootdir=col_values_rootdir_tmp, mode='w')
                    carray_values.flush()
                    shutil.rmtree(col_values_rootdir, ignore_errors=True)
                    shutil.move(col_values_rootdir_tmp, col_values_rootdir)
                else:
                    shutil.rmtree(col_factor_rootdir_tmp, ignore_errors=True)

    def unique(self, col_or_col_list):
        """
        Return a list of unique values of a column or a list of lists of column list

        :param col_or_col_list: a column or a list of columns
        :return:
        """

        if isinstance(col_or_col_list, list):
            col_is_list = True
            col_list = col_or_col_list
        else:
            col_is_list = False
            col_list = [col_or_col_list]

        output = []

        for col in col_list:

            if self.auto_cache or self.cache_valid(col):
                # create factorization cache
                if not self.cache_valid(col):
                    self.cache_factor([col])

                # retrieve values from existing disk-based factorization
                col_values_rootdir = self[col].rootdir + '.values'
                carray_values = bcolz.carray(rootdir=col_values_rootdir, mode='r')
                values = list(carray_values)
            else:
                # factorize on-the-fly
                _, values = ctable_ext.factorize(self[col])
                values = values.values()

            output.append(values)

        if not col_is_list:
            output = output[0]

        return output

    def aggregate_groups(self, ct_agg, nr_groups, skip_key,
                         carray_factor, groupby_cols, agg_ops,
                         dtype_dict, bool_arr=None):
        '''Perform aggregation and place the result in the given ctable.

        Args:
            ct_agg (ctable): the table to hold the aggregation
            nr_groups (int): the number of groups (number of rows in output table)
            skip_key (int): index of the output row to remove from results (used for filtering)
            carray_factor: the carray for each row in the table a reference to the the unique group index
            groupby_cols: the list of 'dimension' columns that are used to perform the groupby over
            output_agg_ops (list): list of tuples of the form: (input_col, agg_op)
                    input_col (string): name of the column to act on
                    agg_op (int): aggregation operation to perform
            bool_arr: a boolean array containing the filter

        '''

        # this creates the groupby columns
        for col in groupby_cols:

            result_array = ctable_ext.groupby_value(self[col], carray_factor,
                                                    nr_groups, skip_key)

            if bool_arr is not None:
                result_array = np.delete(result_array, skip_key)

            ct_agg.addcol(result_array, name=col)
            del result_array

        # this creates the aggregation columns
        for input_col_name, output_col_name, agg_op in agg_ops:

            input_col = self[input_col_name]
            output_col_dtype = dtype_dict[output_col_name]

            input_buffer = np.empty(input_col.chunklen, dtype=input_col.dtype)
            output_buffer = np.zeros(nr_groups, dtype=output_col_dtype)

            if agg_op == 'sum':
                ctable_ext.aggregate_sum(input_col, carray_factor, nr_groups,
                                     skip_key, input_buffer, output_buffer)
            elif agg_op == 'mean':
                ctable_ext.aggregate_mean(input_col, carray_factor, nr_groups,
                                     skip_key, input_buffer, output_buffer)
            elif agg_op == 'std':
                ctable_ext.aggregate_std(input_col, carray_factor, nr_groups,
                                     skip_key, input_buffer, output_buffer)
            elif agg_op == 'count':
                ctable_ext.aggregate_count(input_col, carray_factor, nr_groups,
                                           skip_key, input_buffer, output_buffer)
            elif agg_op == 'count_distinct':
                ctable_ext.aggregate_count_distinct(input_col, carray_factor, nr_groups,
                                     skip_key, input_buffer, output_buffer)
            elif agg_op == 'sorted_count_distinct':
                ctable_ext.aggregate_sorted_count_distinct(input_col, carray_factor, nr_groups,
                                     skip_key, input_buffer, output_buffer)
            else:
                raise KeyError('Unknown aggregation operation ' + str(agg_op))

            if bool_arr is not None:
                output_buffer = np.delete(output_buffer, skip_key)

            ct_agg.addcol(output_buffer, name=output_col_name)
            del output_buffer

        ct_agg.delcol('tmp_col_bquery__')

    def groupby(self, groupby_cols, agg_list, bool_arr=None, rootdir=None):
        """

        Aggregate the ctable

        groupby_cols: a list of columns to groupby over
        agg_list: the aggregation operations, which can be:
         - a list of column names (output has same name and sum is performed)
           ['m1', 'm2', ...]
         - a list of lists, each list contains input column name and operation
           [['m1', 'sum'], ['m2', 'mean'], ...]
         - a list of lists, each list contains input column name, operation and
           output column name
           [['m1', 'sum', 'm1_sum'], ['m1', 'mean', 'm1_mean'], ...]

        Currently supported aggregation operations are:
            - 'sum'
            - 'count'
            - 'count_na'
            - 'count_distinct'
            - 'sorted_count_distinct', data should have been
                  previously presorted
            - 'mean', arithmetic mean (average)
            - 'std', standard deviation

        boolarr: to be added (filtering the groupby factorization input)
        rootdir: the aggregation ctable rootdir
        """

        carray_factor, nr_groups, skip_key = \
            self.make_group_index(groupby_cols, bool_arr)

        # check if the bool_arr actually filters
        if bool_arr is not None and np.all(bool_arr):
            bool_arr = None

        if bool_arr is None:
            expectedlen = nr_groups
        else:
            expectedlen = nr_groups - 1

        ct_agg, dtype_dict, agg_ops = \
            self.create_agg_ctable(groupby_cols, agg_list, expectedlen, rootdir)

        # perform aggregation
        self.aggregate_groups(ct_agg, nr_groups, skip_key,
                              carray_factor, groupby_cols,
                              agg_ops, dtype_dict,
                              bool_arr=bool_arr)

        return ct_agg

    # groupby helper functions
    def factorize_groupby_cols(self, groupby_cols):
        """
        factorizes all columns that are used in the groupby
        it will use cache carrays if available
        if not yet auto_cache is valid, it will create cache carrays

        """
        # first check if the factorized arrays already exist
        # unless we need to refresh the cache
        factor_list = []
        values_list = []

        # factorize the groupby columns
        for col in groupby_cols:

            if self.auto_cache or self.cache_valid(col):
                # create factorization cache if needed
                if not self.cache_valid(col):
                    self.cache_factor([col])

                col_rootdir = self[col].rootdir
                col_factor_rootdir = col_rootdir + '.factor'
                col_values_rootdir = col_rootdir + '.values'
                col_carray_factor = \
                    bcolz.carray(rootdir=col_factor_rootdir, mode='r')
                col_carray_values = \
                    bcolz.carray(rootdir=col_values_rootdir, mode='r')
            else:
                col_carray_factor, values = ctable_ext.factorize(self[col])
                col_carray_values = \
                    bcolz.carray(np.fromiter(values.values(), dtype=self[col].dtype))

            factor_list.append(col_carray_factor)
            values_list.append(col_carray_values)

        return factor_list, values_list

    @staticmethod
    def _int_array_hash(input_list):
        """
        A function to calculate a hash value of multiple integer values, not used at the moment

        Parameters
        ----------
        input_list

        Returns
        -------

        """

        list_len = len(input_list)
        arr_len = len(input_list[0])
        mult_arr = np.full(arr_len, 1000003, dtype=np.long)
        value_arr = np.full(arr_len, 0x345678, dtype=np.long)

        for i, current_arr in enumerate(input_list):
            index = list_len - i - 1
            value_arr ^= current_arr
            value_arr *= mult_arr
            mult_arr += (82520 + index + index)

        value_arr += 97531
        result_carray = bcolz.carray(value_arr)
        del value_arr
        return result_carray

    def create_group_column_factor(self, factor_list, groupby_cols, cache=False, override_cache=False):
        """
        Create a unique, factorized column out of a lost of individual columns

        Parameters
        ----------
        factor_list
        groupby_cols
        cache

        Returns
        -------

        """
        if cache:
            col_rootdir = os.path.join(self.rootdir, self.create_group_base_name(groupby_cols))
            col_factor_rootdir = col_rootdir + '.factor'
            col_factor_rootdir_tmp = tempfile.mkdtemp(prefix='bcolz-')
            col_values_rootdir = col_rootdir + '.values'
            col_values_rootdir_tmp = tempfile.mkdtemp(prefix='bcolz-')
            input_rootdir = tempfile.mkdtemp(prefix='bcolz-')
        else:
            input_rootdir = None
            col_factor_rootdir_tmp = None
            col_values_rootdir_tmp = None

        group_array = bcolz.zeros(0, dtype=np.int64, expectedlen=len(self), rootdir=input_rootdir, mode='w')
        factor_table = bcolz.ctable(factor_list, names=groupby_cols)
        ctable_iter = factor_table.iter(outcols=groupby_cols, out_flavor=tuple)
        ctable_ext.create_group_index(ctable_iter, len(groupby_cols), group_array)

        # now factorize the results
        carray_factor = \
            bcolz.carray([], dtype='int64', expectedlen=self.size, rootdir=col_factor_rootdir_tmp, mode='w')
        carray_factor, values = ctable_ext.factorize(group_array, labels=carray_factor)
        carray_factor.flush()

        if cache and not override_cache and self.group_cache_valid(groupby_cols):
            # another process created the cache in the mean time
            # there is a group cache that we can use, so we remove our work and use the existing
            shutil.rmtree(col_factor_rootdir_tmp, ignore_errors=True)
            carray_factor = bcolz.carray(rootdir=col_factor_rootdir, mode='r')
            carray_values = bcolz.carray(rootdir=col_values_rootdir, mode='r')
        else:
            if cache:
                shutil.rmtree(col_factor_rootdir, ignore_errors=True)
                shutil.move(col_factor_rootdir_tmp, col_factor_rootdir)
                carray_factor = bcolz.carray(rootdir=col_factor_rootdir, mode='r')

            carray_values = \
                bcolz.carray(np.fromiter(values.values(), dtype=np.int64), rootdir=col_values_rootdir_tmp, mode='w')
            carray_values.flush()

            if cache:
                shutil.rmtree(col_values_rootdir, ignore_errors=True)
                shutil.move(col_values_rootdir_tmp, col_values_rootdir)
                carray_values = bcolz.carray(rootdir=col_values_rootdir, mode='r')

        del group_array
        if cache:
            # clean up the temporary file
            shutil.rmtree(input_rootdir, ignore_errors=True)

        return carray_factor, carray_values

    def make_group_index(self, groupby_cols, bool_arr):
        '''Create unique groups for groupby loop

            Args:
                factor_list:
                values_list:
                groupby_cols:
                array_length:
                bool_arr:

            Returns:
                carray: (carray_factor)
                int: (nr_groups) the number of resulting groups
                int: (skip_key)
        '''
        array_length = len(self)
        factor_list, values_list = self.factorize_groupby_cols(groupby_cols)

        # create unique groups for groupby loop
        if len(factor_list) == 0:
            # no columns to groupby over, so directly aggregate the measure
            # columns to 1 total (index 0/zero)
            carray_factor = bcolz.zeros(array_length, dtype='int64')
            carray_values = ['Total']
        elif len(factor_list) == 1:
            # single column groupby, the groupby output column
            # here is 1:1 to the values
            carray_factor = factor_list[0]
            carray_values = values_list[0]
        else:
            # multi column groupby
            # first combine the factorized columns to single values
            if self.group_cache_valid(col_list=groupby_cols):
                # there is a group cache that we can use
                col_rootdir = os.path.join(self.rootdir, self.create_group_base_name(groupby_cols))
                col_factor_rootdir = col_rootdir + '.factor'
                carray_factor = bcolz.carray(rootdir=col_factor_rootdir)
                col_values_rootdir = col_rootdir + '.values'
                carray_values = bcolz.carray(rootdir=col_values_rootdir)
            elif self.auto_cache:
                # create a new group cache
                carray_factor, carray_values = \
                    self.create_group_column_factor(factor_list, groupby_cols, cache=True)
            else:
                # no cached group indexes
                # create a new group cache
                carray_factor, carray_values = \
                    self.create_group_column_factor(factor_list, groupby_cols, cache=False)

        nr_groups = len(carray_values)
        skip_key = None

        if bool_arr is not None:
            # make all non relevant combinations -1
            carray_factor = bcolz.eval(
                '(factor + 1) * bool - 1',
                user_dict={'factor': carray_factor, 'bool': bool_arr})
            # now check how many unique values there are left
            carray_factor, values = ctable_ext.factorize(carray_factor)
            # values might contain one value too much (-1) (no direct lookup
            # possible because values is a reversed dict)
            filter_check = \
                [key for key, value in values.items() if value == -1]
            if filter_check:
                skip_key = filter_check[0]
            # the new nr of groups depends on the outcome after filtering
            nr_groups = len(values)

        # using nr_groups as a total length might be one one off due to the skip_key
        # (skipping a row in aggregation)
        # but that is okay normally

        if skip_key is None:
            # if we shouldn't skip a row, set it at the first row after the total number of groups
            skip_key = nr_groups

        return carray_factor, nr_groups, skip_key

    def create_agg_ctable(self, groupby_cols, agg_list, expectedlen, rootdir):
        '''Create a container for the output table, a dictionary describing it's
            columns and a list of tuples describing aggregation
            operations to perform.

        Args:
            groupby_cols (list): a list of columns to groupby over
            agg_list (list): the aggregation operations (see groupby for more info)
            expectedlen (int): expected length of output table
            rootdir (string): the directory to write the table to

        Returns:
            ctable: A table in the correct format for containing the output of
                    the specified aggregation operations.
            dict: (dtype_dict) dictionary describing columns to create
            list: (agg_ops) list of tuples of the form:
                    (input_col_name, output_col_name, agg_op)
                    input_col_name (string): name of the column to act on
                    output_col_name (string): name of the column to output to
                    agg_op (int): aggregation operation to perform
        '''
        dtype_dict = {}

        # include all the groupby columns
        for col in groupby_cols:
            dtype_dict[col] = self[col].dtype

        agg_ops_list = ['sum', 'count', 'count_distinct', 'sorted_count_distinct', 'mean', 'std']
        agg_ops = []

        for agg_info in agg_list:

            if not isinstance(agg_info, list):
                # example: ['m1', 'm2', ...]
                # default operation (sum) and default output column name (same is input)
                output_col_name = agg_info
                input_col_name = agg_info
                agg_op = 'sum'
            else:
                input_col_name = agg_info[0]
                agg_op = agg_info[1]

                if len(agg_info) == 2:
                    # example: [['m1', 'sum'], ['m2', 'mean], ...]
                    # default output column name
                    output_col_name = input_col_name
                else:
                    # example: [['m1', 'sum', 'mnew1'], ['m1, 'mean','mnew2'], ...]
                    # fully specified
                    output_col_name = agg_info[2]

            if agg_op not in agg_ops_list:
                raise NotImplementedError(
                    'Unknown Aggregation Type: ' + str(agg_op))

            # choose output column dtype based on aggregation operation and
            # input column dtype
            # TODO: check if the aggregation columns is numeric
            # NB: we could build a concatenation for strings like pandas, but I would really prefer to see that as a
            # separate operation
            if agg_op in ('count', 'count_distinct', 'sorted_count_distinct'):
                output_col_dtype = np.dtype(np.int64)
            elif agg_op in ('mean', 'std'):
                output_col_dtype = np.dtype(np.float64)
            else:
                output_col_dtype = self[input_col_name].dtype

            dtype_dict[output_col_name] = output_col_dtype

            # save output
            agg_ops.append((input_col_name, output_col_name, agg_op))

        # create aggregation table
        ct_agg = bcolz.ctable(
            np.zeros(expectedlen, [('tmp_col_bquery__', np.bool)]),
            expectedlen=expectedlen,
            rootdir=rootdir)

        return ct_agg, dtype_dict, agg_ops

    def where_terms(self, term_list):
        """
        Create a boolean array where `term_list` is true.
        A terms list has a [(col, operator, value), ..] construction.
        Eg. [('sales', '>', 2), ('state', 'in', ['IL', 'AR'])]

        :param term_list:
        :param outcols:
        :param limit:
        :param skip:
        :return: :raise ValueError:
        """

        if type(term_list) not in [list, set, tuple]:
            raise ValueError("Only term lists are supported")

        col_list = []
        op_list = []
        value_list = []

        for term in term_list:
            # get terms
            filter_col = term[0]
            filter_operator = term[1].lower().strip(' ')
            filter_value = term[2]

            # check values
            if filter_col not in self.cols:
                raise KeyError(unicode(filter_col) + ' not in table')

            if filter_operator in ['==', 'eq']:
                op_id = 1
            elif filter_operator in ['!=', 'neq']:
                op_id = 2
            elif filter_operator in ['in']:
                op_id = 3
            elif filter_operator in ['nin', 'not in']:
                op_id = 4
            elif filter_operator in ['>']:
                op_id = 5
            elif filter_operator in ['>=']:
                op_id = 6
            elif filter_operator in ['<']:
                op_id = 7
            elif filter_operator in ['<=']:
                op_id = 8
            else:
                raise KeyError(unicode(filter_operator) + ' is not an accepted operator for filtering')

            if op_id in [3, 4]:
                if type(filter_value) not in [list, set, tuple]:
                    raise ValueError("In selections need lists, sets or tuples")

                if len(filter_value) < 1:
                    raise ValueError("A value list needs to have values")

                # optimize lists of 1 value
                if len(filter_value) == 1:
                    if op_id == 3:
                        op_id = 1
                    else:
                        op_id = 2

                    filter_value = filter_value[0]
                else:
                    filter_value = set(filter_value)

            # prepare input for filter creation
            col_list.append(filter_col)
            op_list.append(op_id)
            value_list.append(filter_value)

        # create boolean array and fill it
        boolarr = bcolz.carray(np.ones(0, dtype=np.bool), expectedlen=self.len)
        ctable_iter = self[col_list].iter(out_flavor='tuple')
        ctable_ext.apply_where_terms(ctable_iter, op_list, value_list, boolarr)

        return boolarr

    def is_in_ordered_subgroups(self, basket_col=None, bool_arr=None,
                                _max_len_subgroup=1000):
        """"""
        assert basket_col is not None

        if bool_arr is None:
            bool_arr = bcolz.ones(self.len)

        return \
            ctable_ext.is_in_ordered_subgroups(
                self[basket_col], bool_arr=bool_arr,
                _max_len_subgroup=_max_len_subgroup)

