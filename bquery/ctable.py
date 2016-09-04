# internal imports
from bquery import ctable_ext

# external imports
import numpy as np
import bcolz
import os
from bquery.ctable_ext import \
    SUM, COUNT, COUNT_NA, COUNT_DISTINCT, SORTED_COUNT_DISTINCT, \
    MEAN, STDEV


class ctable(bcolz.ctable):
    def cache_valid(self, col):
        """
        Checks whether the column has a factorization that exists and is not older than the source

        :param col:
        :return:
        """
        if self.rootdir:
            col_org_file_check = self[col].rootdir + '/__attrs__'
            col_values_file_check = self[col].rootdir + '.values/__attrs__'

            if not os.path.exists(col_org_file_check):
                raise KeyError(str(col) + ' does not exist')

            if os.path.exists(col_values_file_check):
                return os.path.getctime(col_org_file_check) < os.path.getctime(col_values_file_check)
            else:
                return False
        else:
            return False

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

        for col in col_list:

            # create cache if needed
            if refresh or not self.cache_valid(col):
                col_rootdir = self[col].rootdir
                col_factor_rootdir = col_rootdir + '.factor'
                col_values_rootdir = col_rootdir + '.values'

                carray_factor = \
                    bcolz.carray([], dtype='int64', expectedlen=self.size,
                                 rootdir=col_factor_rootdir, mode='w')
                _, values = \
                    ctable_ext.factorize(self[col], labels=carray_factor)
                carray_factor.flush()

                carray_values = \
                    bcolz.carray(np.fromiter(values.values(), dtype=self[col].dtype),
                                 rootdir=col_values_rootdir, mode='w')
                carray_values.flush()

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

            if self.cache_valid(col):
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
                         factor_carray, groupby_cols, output_agg_ops,
                         dtype_dict, bool_arr=None):
        '''Perform aggregation and place the result in the given ctable.

        Args:
            ct_agg (ctable): the table to hold the aggregation
            nr_groups (int): the number of groups (number of rows in output table)
            skip_key (int): index of the output row to remove from results (used for filtering)
            factor_carray: the carray for each row in the table a reference to the the unique group index
            groupby_cols: the list of 'dimension' columns that are used to perform the groupby over
            output_agg_ops (list): list of tuples of the form: (input_col, agg_op)
                    input_col (string): name of the column to act on
                    agg_op (int): aggregation operation to perform
            bool_arr: a boolean array containing the filter

        '''

        # this creates the groupby columns
        for col in groupby_cols:

            result_array = ctable_ext.groupby_value(self[col], factor_carray,
                                                    nr_groups, skip_key)

            if bool_arr is not None:
                result_array = np.delete(result_array, skip_key)

            ct_agg.addcol(result_array, name=col)
            del result_array

        # this creates the aggregation columns
        for input_col_name, output_col_name, agg_op in output_agg_ops:

            input_col = self[input_col_name]
            output_col_dtype = dtype_dict[output_col_name]

            input_buffer = np.empty(input_col.chunklen, dtype=input_col.dtype)
            output_buffer = np.zeros(nr_groups, dtype=output_col_dtype)

            try:
                ctable_ext.aggregate(input_col, factor_carray, nr_groups,
                                     skip_key, input_buffer, output_buffer,
                                     agg_op)
            except TypeError:
                raise NotImplementedError(
                    'Column dtype ({0}) not supported for aggregation yet '
                    '(only int32, int64 & float64)'.format(str(input_col.dtype)))
            except Exception as e:
                raise e

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

        if not agg_list:
            raise AttributeError('One or more aggregation operations '
                                 'need to be defined')

        factor_list, values_list = self.factorize_groupby_cols(groupby_cols)

        factor_carray, nr_groups, skip_key = \
            self.make_group_index(factor_list, values_list, groupby_cols,
                                  len(self), bool_arr)

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
                              factor_carray, groupby_cols,
                              agg_ops, dtype_dict,
                              bool_arr=bool_arr)

        return ct_agg

    # groupby helper functions
    def factorize_groupby_cols(self, groupby_cols):
        """

        :type self: ctable
        """
        # first check if the factorized arrays already exist
        # unless we need to refresh the cache
        factor_list = []
        values_list = []

        # factorize the groupby columns
        for col in groupby_cols:

            if self.cache_valid(col):
                col_rootdir = self[col].rootdir
                col_factor_rootdir = col_rootdir + '.factor'
                col_values_rootdir = col_rootdir + '.values'
                col_factor_carray = \
                    bcolz.carray(rootdir=col_factor_rootdir, mode='r')
                col_values_carray = \
                    bcolz.carray(rootdir=col_values_rootdir, mode='r')
            else:
                col_factor_carray, values = ctable_ext.factorize(self[col])
                col_values_carray = \
                    bcolz.carray(np.fromiter(values.values(), dtype=self[col].dtype))

            factor_list.append(col_factor_carray)
            values_list.append(col_values_carray)

        return factor_list, values_list

    def make_group_index(self, factor_list, values_list, groupby_cols,
                         array_length, bool_arr):
        '''Create unique groups for groupby loop

            Args:
                factor_list:
                values_list:
                groupby_cols:
                array_length:
                bool_arr:

            Returns:
                carray: (factor_carray)
                int: (nr_groups) the number of resulting groups
                int: (skip_key)
        '''

        def _int_array_hash(input_list):

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

        # create unique groups for groupby loop
        if len(factor_list) == 0:
            # no columns to groupby over, so directly aggregate the measure
            # columns to 1 total (index 0/zero)
            factor_carray = bcolz.zeros(array_length, dtype='int64')
            values = ['Total']
        elif len(factor_list) == 1:
            # single column groupby, the groupby output column
            # here is 1:1 to the values
            factor_carray = factor_list[0]
            values = values_list[0]
        else:
            # multi column groupby
            # todo: this might also be cached in the future
            # todo: move out-of-core instead of a numpy array
            # first combine the factorized columns to single values
            group_array = _int_array_hash(factor_list)
            factor_carray, values = ctable_ext.factorize(group_array)

        skip_key = None

        if bool_arr is not None:
            # make all non relevant combinations -1
            factor_carray = bcolz.eval(
                '(factor + 1) * bool - 1',
                user_dict={'factor': factor_carray, 'bool': bool_arr})
            # now check how many unique values there are left
            factor_carray, values = ctable_ext.factorize(factor_carray)
            # values might contain one value too much (-1) (no direct lookup
            # possible because values is a reversed dict)
            filter_check = \
                [key for key, value in values.items() if value == -1]
            if filter_check:
                skip_key = filter_check[0]

        # using nr_groups as a total length might be one one off due to the skip_key
        # (skipping a row in aggregation)
        # but that is okay normally
        nr_groups = len(values)
        if skip_key is None:
            # if we shouldn't skip a row, set it at the first row after the total number of groups
            skip_key = nr_groups

        return factor_carray, nr_groups, skip_key

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

        agg_ops = []
        op_translation = {
            'sum': SUM,
            'count': COUNT,
            'count_na': COUNT_NA,
            'count_distinct': COUNT_DISTINCT,
            'sorted_count_distinct': SORTED_COUNT_DISTINCT,
            'mean': MEAN,
            'std': STDEV
        }

        for agg_info in agg_list:

            if not isinstance(agg_info, list):
                # example: ['m1', 'm2', ...]
                # default operation (sum) and default output column name (same is input)
                output_col_name = agg_info
                input_col_name = agg_info
                agg_op = SUM
            else:
                input_col_name = agg_info[0]
                agg_op_input = agg_info[1]

                if len(agg_info) == 2:
                    # example: [['m1', 'sum'], ['m2', 'mean], ...]
                    # default output column name
                    output_col_name = input_col_name
                else:
                    # example: [['m1', 'sum', 'mnew1'], ['m1, 'mean','mnew2'], ...]
                    # fully specified
                    output_col_name = agg_info[2]
                if agg_op_input not in op_translation:
                    raise NotImplementedError(
                        'Unknown Aggregation Type: ' + unicode(agg_op_input))
                agg_op = op_translation[agg_op_input]


            # choose output column dtype based on aggregation operation and
            # input column dtype
            # TODO: check if the aggregation columns is numeric
            # NB: we could build a concatenation for strings like pandas, but I would really prefer to see that as a
            # separate operation
            if agg_op in (COUNT, COUNT_NA, COUNT_DISTINCT, SORTED_COUNT_DISTINCT):
                output_col_dtype = np.dtype(np.int64)
            elif agg_op in (MEAN, STDEV):
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
        TEMPORARY WORKAROUND TILL NUMEXPR WORKS WITH IN
        where_terms(term_list, outcols=None, limit=None, skip=0)

        Iterate over rows where `term_list` is true.
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

        eval_string = ''
        eval_list = []

        for term in term_list:
            filter_col = term[0]
            filter_operator = term[1].lower()
            filter_value = term[2]

            if filter_operator not in ['in', 'not in']:
                # direct filters should be added to the eval_string

                # add and logic if not the first term
                if eval_string:
                    eval_string += ' & '

                eval_string += '(' + filter_col + ' ' \
                               + filter_operator + ' ' \
                               + str(filter_value) + ')'

            elif filter_operator in ['in', 'not in']:
                # Check input
                if type(filter_value) not in [list, set, tuple]:
                    raise ValueError("In selections need lists, sets or tuples")

                if len(filter_value) < 1:
                    raise ValueError("A value list needs to have values")

                elif len(filter_value) == 1:
                    # handle as eval
                    # add and logic if not the first term
                    if eval_string:
                        eval_string += ' & '

                    if filter_operator == 'not in':
                        filter_operator = '!='
                    else:
                        filter_operator = '=='

                    eval_string += '(' + filter_col + ' ' + \
                                   filter_operator

                    filter_value = filter_value[0]

                    if type(filter_value) == str:
                        filter_value = '"' + filter_value + '"'
                    else:
                        filter_value = str(filter_value)

                    eval_string += filter_value + ') '

                else:

                    if type(filter_value) in [list, tuple]:
                        filter_value = set(filter_value)

                    eval_list.append(
                        (filter_col, filter_operator, filter_value)
                    )
            else:
                raise ValueError(
                    "Input not correctly formatted for eval or list filtering"
                )

        # (1) Evaluate terms in eval
        # return eval_string, eval_list
        if eval_string:
            boolarr = self.eval(eval_string)
            if eval_list:
                # convert to numpy array for array_is_in
                boolarr = boolarr[:]
        else:
            boolarr = np.ones(self.size, dtype=bool)

        # (2) Evaluate other terms like 'in' or 'not in' ...
        for term in eval_list:

            name = term[0]
            col = self.cols[name]

            operator = term[1]
            if operator.lower() == 'not in':
                reverse = True
            elif operator.lower() == 'in':
                reverse = False
            else:
                raise ValueError(
                    "Input not correctly formatted for list filtering"
                )

            value_set = set(term[2])

            ctable_ext.carray_is_in(col, value_set, boolarr, reverse)

        if eval_list:
            # convert boolarr back to carray
            boolarr = bcolz.carray(boolarr)

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
