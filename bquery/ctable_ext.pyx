import numpy as np
cimport numpy as np

import cython
import bcolz as bz
from bcolz.carray_ext cimport carray, chunk

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

from libc.stdlib cimport malloc
from libc.string cimport strcpy
from khash cimport *

# ----------------------------------------------------------------------------
#                        GLOBAL DEFINITIONS
# ----------------------------------------------------------------------------

SUM = 0
DEF _SUM = 0

MEAN = 5
DEF _MEAN = 5

STDEV = 6
DEF _STDEV = 6

COUNT = 1
DEF _COUNT = 1

COUNT_NA = 2
DEF _COUNT_NA = 2

COUNT_DISTINCT = 3
DEF _COUNT_DISTINCT = 3

SORTED_COUNT_DISTINCT = 4
DEF _SORTED_COUNT_DISTINCT = 4
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
#                        FUSED TYPES
# ----------------------------------------------------------------------------
# fused types (templating) from
# http://docs.cython.org/src/userguide/fusedtypes.html
ctypedef fused numpy_native_number_input:
    np.int64_t
    np.int32_t
    np.float64_t

ctypedef fused numpy_native_number_output:
    np.int64_t
    np.int32_t
    np.float64_t


# ----------------------------------------------------------------------------

# Factorize Section
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_str_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       np.ndarray in_buffer,
                       np.ndarray[np.uint64_t] out_buffer,
                       kh_str_t *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):
    cdef:
        Py_ssize_t i, idx
        int ret
        char * element
        char * insert
        khiter_t k

    ret = 0

    for i in range(iter_range):
        # TODO: Consider indexing directly into the array for efficiency
        element = in_buffer[i]
        k = kh_get_str(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            # allocate enough memory to hold the string, add one for the
            # null byte that marks the end of the string.
            insert = <char *>malloc(allocation_size)
            # TODO: is strcpy really the best way to copy a string?
            strcpy(insert, element)
            k = kh_put_str(table, insert, &ret)
            table.vals[k] = idx = count[0]
            reverse[count[0]] = element
            count[0] += 1
        out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_str(carray carray_, carray labels=None):
    cdef:
        Py_ssize_t len_carray, count, chunklen, len_in_buffer
        dict reverse
        np.ndarray in_buffer
        np.ndarray[np.uint64_t] out_buffer
        kh_str_t *table

    count = 0
    ret = 0
    reverse = {}

    len_carray = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=len_carray)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    in_buffer = np.empty(chunklen, dtype=carray_.dtype)
    table = kh_init_str()

    for in_buffer in bz.iterblocks(carray_):
        len_in_buffer = len(in_buffer)

        _factorize_str_helper(len_in_buffer,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )

        # TODO: need to use an explicit loop here for speed?
        labels.append(out_buffer[:len_in_buffer].astype(np.int64))

    kh_destroy_str(table)

    return labels, reverse

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_number_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       np.ndarray[numpy_native_number_input] in_buffer,
                       np.ndarray[np.uint64_t] out_buffer,
                       void *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):

    cdef:
        Py_ssize_t i, idx
        int ret
        numpy_native_number_input element
        khiter_t
        kh_int64_t *int64_table
        kh_int32_t *int32_table
        kh_float64_t *float64_table

    ret = 0

    # Only one of these branches should be compiled for each specialization
    # see here for info on branches with fused types:
    # http://docs.cython.org/src/userguide/fusedtypes.html#type-checking-specializations
    if numpy_native_number_input is np.int64_t:
        int64_table = <kh_int64_t*> table

        for i in range(iter_range):
            # TODO: Consider indexing directly into the array for efficiency
            element = in_buffer[i]
            k = kh_get_int64(int64_table, element)
            if k != int64_table.n_buckets:
                idx = int64_table.vals[k]
            else:
                k = kh_put_int64(int64_table, element, &ret)
                int64_table.vals[k] = idx = count[0]
                reverse[count[0]] = element
                count[0] += 1
            out_buffer[i] = idx

    elif numpy_native_number_input is np.int32_t:
        int32_table = <kh_int32_t*> table

        for i in range(iter_range):
            element = in_buffer[i]
            k = kh_get_int32(int32_table, element)
            if k != int32_table.n_buckets:
                idx = int32_table.vals[k]
            else:
                k = kh_put_int32(int32_table, element, &ret)
                int32_table.vals[k] = idx = count[0]
                reverse[count[0]] = element
                count[0] += 1
            out_buffer[i] = idx

    elif numpy_native_number_input is np.float64_t:
        float64_table = <kh_float64_t*> table

        for i in range(iter_range):
            element = in_buffer[i]
            k = kh_get_float64(float64_table, element)
            if k != float64_table.n_buckets:
                idx = float64_table.vals[k]
            else:
                k = kh_put_float64(float64_table, element, &ret)
                float64_table.vals[k] = idx = count[0]
                reverse[count[0]] = element
                count[0] += 1
            out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
cdef factorize_number(carray carray_,  numpy_native_number_input typehint, carray labels=None):
    # fused type conversion
    if numpy_native_number_input is np.int64_t:
        p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
        p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
        p_dtype = np.float64

    cdef:
        Py_ssize_t len_carray, i, count, chunklen, len_in_buffer
        dict reverse
        np.ndarray[numpy_native_number_input] in_buffer
        np.ndarray[np.uint64_t] out_buffer
        void *table

    count = 0
    reverse = {}

    len_carray = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=len_carray)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')

    if numpy_native_number_input is np.int64_t:
        table = kh_init_int64()
    elif numpy_native_number_input is np.int32_t:
        table = kh_init_int32()
    elif numpy_native_number_input is np.float64_t:
        table = kh_init_float64()
    else:
        table = kh_init_int64()

    for in_buffer in bz.iterblocks(carray_):
        len_in_buffer = len(in_buffer)
        _factorize_number_helper[numpy_native_number_input](len_in_buffer,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        <void*> table,
                        &count,
                        reverse,
                        )


        # TODO: need to use an explicit loop here for speed?
        # compress out_buffer into labels
        labels.append(out_buffer[:len_in_buffer].astype(np.int64))

    if numpy_native_number_input is np.int64_t:
        kh_destroy_int64(<kh_int64_t*>table)
    elif numpy_native_number_input is np.int32_t:
        kh_destroy_int32(<kh_int32_t*>table)
    elif numpy_native_number_input is np.float64_t:
        kh_destroy_float64(<kh_float64_t*>table)

    return labels, reverse

cpdef factorize(carray carray_, carray labels=None):
    cdef:
        np.int64_t hint64 = 0
        np.int32_t hint32 = 0
        np.float64_t hfloat64 = 0

    if carray_.dtype == 'int32':
        labels, reverse = factorize_number(carray_, hint32, labels=labels)
    elif carray_.dtype == 'int64':
        labels, reverse = factorize_number(carray_, hint64, labels=labels)
    elif carray_.dtype == 'float64':
        labels, reverse = factorize_number(carray_, hfloat64, labels=labels)
    else:
        #TODO: check that the input is a string_ dtype type
        labels, reverse = factorize_str(carray_, labels=labels)
    return labels, reverse

# ---------------------------------------------------------------------------
# Aggregation Section
@cython.boundscheck(False)
@cython.wraparound(False)
def groupsort_indexer(carray index, Py_ssize_t ngroups):
    cdef:
        Py_ssize_t label, n, i, len_in_buffer
        np.ndarray[int64_t] counts, where, np_result
        # --
        carray c_result
        Py_ssize_t index_chunk_nr, index_chunk_len, leftover_elements

        np.ndarray[int64_t] in_buffer

    index_chunk_len = index.chunklen
    in_buffer = np.empty(index_chunk_len, dtype='int64')
    index_chunk_nr = 0

    # count group sizes, location 0 for NA
    counts = np.zeros(ngroups + 1, dtype=np.int64)
    n = len(index)

    for in_buffer in bz.iterblocks(index):
        len_in_buffer = len(in_buffer)

        for i in range(len_in_buffer):
            counts[index[i] + 1] += 1

    # mark the start of each contiguous group of like-indexed data
    where = np.zeros(ngroups + 1, dtype=np.int64)
    for i from 1 <= i < ngroups + 1:
        where[i] = where[i - 1] + counts[i - 1]

    # this is our indexer
    np_result = np.zeros(n, dtype=np.int64)
    for i from 0 <= i < n:
        label = index[i] + 1
        np_result[where[label]] = i
        where[label] += 1

    return np_result, counts

cdef count_unique(np.ndarray[numpy_native_number_input] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx
        int ret = 0
        numpy_native_number_input val
        khiter_t k
        np.uint64_t count = 0
        bint seen_na = 0
        kh_int64_t *int64_table
        kh_int32_t *int32_table
        kh_float64_t *float64_table

    # Only one of these branches should be compiled for each specialization
    if numpy_native_number_input is np.int64_t:

        int64_table = kh_init_int64()

        for i in range(n):
            val = values[i]

            if val == val:
                k = kh_get_int64(int64_table, val)
                if k == int64_table.n_buckets:
                    k = kh_put_int64(int64_table, val, &ret)
                    count += 1
            elif not seen_na:
                seen_na = 1
                count += 1

        kh_destroy_int64(int64_table)

    elif numpy_native_number_input is np.int32_t:

        int32_table = kh_init_int32()

        for i in range(n):
            val = values[i]

            if val == val:
                k = kh_get_int32(int32_table, val)
                if k == int32_table.n_buckets:
                    k = kh_put_int32(int32_table, val, &ret)
                    count += 1
            elif not seen_na:
                seen_na = 1
                count += 1

        kh_destroy_int32(int32_table)

    elif numpy_native_number_input is np.float64_t:

        float64_table = kh_init_float64()

        for i in range(n):
            val = values[i]

            if val == val:
                k = kh_get_float64(float64_table, val)
                if k == float64_table.n_buckets:
                    k = kh_put_float64(float64_table, val, &ret)
                    count += 1
            elif not seen_na:
                seen_na = 1
                count += 1

        kh_destroy_float64(float64_table)


    return count

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef aggregate(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key,
               np.ndarray[numpy_native_number_input] in_buffer,
               np.ndarray[numpy_native_number_output] out_buffer,
               agg_method):

    # fused type conversion
    if numpy_native_number_input is np.int64_t:
       p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
       p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
       p_dtype = np.float64

    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts

        np.ndarray[np.int64_t] factor_buffer

        np.ndarray[numpy_native_number_input] last_values

        numpy_native_number_input v
        bint count_distinct_started = 0
        carray num_uniques

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)

    if agg_method == _COUNT_DISTINCT:
        positions, counts = groupsort_indexer(ca_factor, nr_groups)
        start_counts = 0
        end_counts = 0
        for j in range(len(counts) - 1):
            start_counts = end_counts
            end_counts = start_counts + counts[j + 1]

            out_buffer[j] = \
                count_unique[numpy_native_number_input](ca_input[positions[start_counts:end_counts]])

        return

    factor_chunk_nr = 0
    factor_buffer = next(iter_ca_factor)
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0

    # create special buffers for complex operations
    if agg_method == _MEAN or agg_method == _STDEV:
        count_buffer = np.zeros(nr_groups, dtype='int64')
    if agg_method == _STDEV:
        mean_buffer = np.zeros(nr_groups, dtype='float64')

    try:
        for in_buffer in bz.iterblocks(ca_input):
            len_in_buffer = len(in_buffer)

            # loop through rows
            for i in range(len_in_buffer):

                # go to next factor buffer if necessary

                if factor_chunk_row == factor_buffer_len:
                    factor_chunk_nr += 1
                    factor_buffer = next(iter_ca_factor)
                    factor_buffer_len = len(factor_buffer)
                    factor_chunk_row = 0

                # retrieve index
                current_index = factor_buffer[factor_chunk_row]
                factor_chunk_row += 1

                # update value if it's not an invalid index
                if current_index != skip_key:
                    if agg_method == _SUM:
                        out_buffer[current_index] += <numpy_native_number_output> in_buffer[i]
                    elif agg_method == _MEAN:
                        # method from Knuth
                        count_buffer[current_index] += 1
                        delta = in_buffer[i] - out_buffer[current_index]
                        out_buffer[current_index] += delta / count_buffer[current_index]
                    elif agg_method == _STDEV:
                        count_buffer[current_index] += 1
                        delta = in_buffer[i] - mean_buffer[current_index]
                        mean_buffer[current_index] += delta / count_buffer[current_index]
                        # M2 = M2 + delta*(x - mean)
                        out_buffer[current_index] += delta * (in_buffer[i] - mean_buffer[current_index])
                    elif agg_method == _COUNT:
                        out_buffer[current_index] += 1
                    elif agg_method == _COUNT_NA:

                        v = in_buffer[i]
                        if v == v:  # skip NA values
                            out_buffer[current_index] += 1
                    elif agg_method == _SORTED_COUNT_DISTINCT:
                        v = in_buffer[i]
                        if not count_distinct_started:
                            count_distinct_started = 1
                            last_values = np.zeros(nr_groups, dtype=p_dtype)
                            last_values[0] = v
                            out_buffer[0] = 1
                        else:
                            if v != last_values[current_index]:
                                out_buffer[current_index] += 1

                        last_values[current_index] = v
                    else:
                        raise NotImplementedError('sumtype not supported')
    except StopIteration:
        pass
    finally:
        del iter_ca_factor

    if agg_method == _STDEV:
        for i in range(len(out_buffer)):
            out_buffer[i] = np.sqrt(out_buffer[i] / (count_buffer[i]))

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef groupby_value(carray ca_input, carray ca_factor, Py_ssize_t nr_groups, Py_ssize_t skip_key):
    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks

        np.ndarray in_buffer
        np.ndarray[np.int64_t] factor_buffer
        np.ndarray out_buffer

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)


    factor_total_chunks = ca_factor.nchunks
    factor_chunk_nr = 0
    factor_buffer = next(iter_ca_factor)
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0

    out_buffer = np.zeros(nr_groups, dtype=ca_input.dtype)

    try:
        for in_buffer in bz.iterblocks(ca_input):
            len_in_buffer = len(in_buffer)

            for i in range(len_in_buffer):

                # go to next factor buffer if necessary
                if factor_chunk_row == factor_buffer_len:
                    factor_chunk_nr += 1
                    factor_buffer = next(iter_ca_factor)
                    factor_buffer_len = len(factor_buffer)
                    factor_chunk_row = 0

                # retrieve index
                current_index = factor_buffer[factor_chunk_row]
                factor_chunk_row += 1

                # update value if it's not an invalid index
                if current_index != skip_key:
                    out_buffer[current_index] = in_buffer[i]
    except StopIteration:
        pass
    finally:
        del iter_ca_factor

    # check whether a row has to be fixed
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

    return out_buffer


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef is_in_ordered_subgroups(carray groups_col, carray bool_arr=None,
                              _max_len_subgroup=1000):
    """
    Mark whole basket containing certain items

    :param groups_col: carray containing ordered groups
    :param bool_arr: bool array showing if an desired item is present
    :param _max_len_subgroup: expected max. basket size
    :return: bool array marking the whole group as True if item found
    """
    cdef:
        np.int64_t previous_item
        np.int64_t actual_item
        Py_ssize_t blen
        Py_ssize_t len_subgroup = 0
        Py_ssize_t max_len_subgroup
        Py_ssize_t n
        np.npy_bool is_in = False
        carray ret
        np.ndarray x, x_ones, x_zeros
        np.ndarray bl_basket, bl_bool_arr

    max_len_subgroup = _max_len_subgroup
    ret = bz.zeros(0, dtype='bool', expectedlen=groups_col.len)
    blen = min([groups_col.chunklen, bool_arr.chunklen])
    previous_item = groups_col[0]

    x_ones = np.ones(max_len_subgroup, dtype='bool')
    x_zeros = np.zeros(max_len_subgroup, dtype='bool')

    for bl_basket, bl_bool_arr in izip(
            bz.iterblocks(groups_col, blen=blen),
            bz.iterblocks(bool_arr, blen=blen)):

        for n in range(len(bl_basket)):

            actual_item = bl_basket[n]

            if previous_item != actual_item:
                if len_subgroup > max_len_subgroup:
                    max_len_subgroup = len_subgroup
                    x_ones = np.ones(max_len_subgroup, dtype='bool')
                    x_zeros = np.zeros(max_len_subgroup, dtype='bool')
                if is_in:
                    x = x_ones[0:len_subgroup]
                else:
                    x = x_zeros[0:len_subgroup]
                ret.append(x)
                # - reset vars -
                is_in = False
                len_subgroup = 0

            if bl_bool_arr[n]:
                is_in = True

            len_subgroup += 1
            previous_item = actual_item

    if len_subgroup > max_len_subgroup:
        max_len_subgroup = len_subgroup
        x_ones = np.ones(max_len_subgroup, dtype='bool')
        x_zeros = np.zeros(max_len_subgroup, dtype='bool')
    if is_in:
        x = x_ones[0:len_subgroup]
    else:
        x = x_zeros[0:len_subgroup]

    ret.append(x)

    return ret


# ---------------------------------------------------------------------------
# Temporary Section
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef carray_is_in(carray col, set value_set, np.ndarray boolarr, bint reverse):
    """
    TEMPORARY WORKAROUND till numexpr support in list operations

    Update a boolean array with checks whether the values of a column (col) are in a set (value_set)
    Reverse means "not in" functionality

    For the 0d array work around, see https://github.com/Blosc/bcolz/issues/61

    :param col:
    :param value_set:
    :param boolarr:
    :param reverse:
    :return:
    """
    cdef Py_ssize_t i
    i = 0
    if not reverse:
        for val in col.iter():
            if val not in value_set:
                boolarr[i] = False
            i += 1
    else:
        for val in col.iter():
            if val in value_set:
                boolarr[i] = False
            i += 1

# Translate existing arrays
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef translate_int64(carray input_, carray output_, dict lookup, np.npy_int64 default=-1):
    """
    used for internal vf functions; creates a factorized array from an existing lookup dictionary

    :param input_:
    :param output_:
    :param lookup:
    :param default:
    :return:
    """
    cdef:
        Py_ssize_t i, chunklen, len_in_buffer
        np.ndarray[np.npy_int64] in_buffer
        np.ndarray[np.npy_int64] out_buffer

    chunklen = input_.chunklen
    out_buffer = np.empty(chunklen, dtype='int64')

    for in_buffer in bz.iterblocks(input_):
        len_in_buffer = len(in_buffer)

        for i in range(len_in_buffer):
            element = in_buffer[i]
            out_buffer[i] = lookup.get(element, default)

        # compress out_buffer into labels
        output_.append(out_buffer[:len_in_buffer].astype(np.int64))
