import numpy as np
from numpy cimport ndarray, dtype, npy_intp, npy_int32, \
    npy_uint64, npy_int64, npy_float64, npy_bool

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

COUNT = 1
DEF _COUNT = 1

COUNT_NA = 2
DEF _COUNT_NA = 2

COUNT_DISTINCT = 3
DEF _COUNT_DISTINCT = 3

SORTED_COUNT_DISTINCT = 4
DEF _SORTED_COUNT_DISTINCT = 4

MEAN = 5
DEF _MEAN = 5

STDEV = 6
DEF _STDEV = 6

# ----------------------------------------------------------------------------

# Factorize Section
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_str_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       ndarray in_buffer,
                       ndarray[npy_uint64] out_buffer,
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
        chunk chunk_
        Py_ssize_t n, i, count, chunklen, leftover_elements
        dict reverse
        ndarray in_buffer
        ndarray[npy_uint64] out_buffer
        kh_str_t *table

    count = 0
    ret = 0
    reverse = {}

    n = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=n)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    in_buffer = np.empty(chunklen, dtype=carray_.dtype)
    table = kh_init_str()

    for i in range(carray_.nchunks):
        chunk_ = carray_.chunks[i]
        # decompress into in_buffer
        chunk_._getitem(0, chunklen, in_buffer.data)
        _factorize_str_helper(chunklen,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer.astype(np.int64))

    leftover_elements = cython.cdiv(carray_.leftover, carray_.atomsize)
    if leftover_elements > 0:
        _factorize_str_helper(leftover_elements,
                          carray_.dtype.itemsize + 1,
                          carray_.leftover_array,
                          out_buffer,
                          table,
                          &count,
                          reverse,
                          )

    # compress out_buffer into labels
    labels.append(out_buffer[:leftover_elements].astype(np.int64))

    kh_destroy_str(table)

    return labels, reverse

{% for factor_type in factor_types -%}
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_{{ factor_type }}_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       ndarray[npy_{{ factor_type }}] in_buffer,
                       ndarray[npy_uint64] out_buffer,
                       kh_{{ factor_type }}_t *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):
    cdef:
        Py_ssize_t i, idx
        int ret
        npy_{{ factor_type }} element
        khiter_t k

    ret = 0

    for i in range(iter_range):
        # TODO: Consider indexing directly into the array for efficiency
        element = in_buffer[i]
        k = kh_get_{{ factor_type }}(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            k = kh_put_{{ factor_type }}(table, element, &ret)
            table.vals[k] = idx = count[0]
            reverse[count[0]] = element
            count[0] += 1
        out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_{{ factor_type }}(carray carray_, carray labels=None):
    cdef:
        chunk chunk_
        Py_ssize_t n, i, count, chunklen, leftover_elements
        dict reverse
        ndarray[npy_{{ factor_type }}] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_{{ factor_type }}_t *table

    count = 0
    ret = 0
    reverse = {}

    n = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=n)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    in_buffer = np.empty(chunklen, dtype='{{ factor_type }}')
    table = kh_init_{{ factor_type }}()

    for i in range(carray_.nchunks):
        chunk_ = carray_.chunks[i]
        # decompress into in_buffer
        chunk_._getitem(0, chunklen, in_buffer.data)
        _factorize_{{ factor_type }}_helper(chunklen,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer.astype(np.int64))

    leftover_elements = cython.cdiv(carray_.leftover, carray_.atomsize)
    if leftover_elements > 0:
        _factorize_{{ factor_type }}_helper(leftover_elements,
                          carray_.dtype.itemsize + 1,
                          carray_.leftover_array,
                          out_buffer,
                          table,
                          &count,
                          reverse,
                          )

    # compress out_buffer into labels
    labels.append(out_buffer[:leftover_elements].astype(np.int64))

    kh_destroy_{{ factor_type }}(table)

    return labels, reverse

{% endfor -%}

cpdef factorize(carray carray_, carray labels=None):
    if carray_.dtype == 'int32':
        labels, reverse = factorize_int32(carray_, labels=labels)
    elif carray_.dtype == 'int64':
        labels, reverse = factorize_int64(carray_, labels=labels)
    elif carray_.dtype == 'float64':
        labels, reverse = factorize_float64(carray_, labels=labels)
    else:
        #TODO: check that the input is a string_ dtype type
        labels, reverse = factorize_str(carray_, labels=labels)
    return labels, reverse

# ---------------------------------------------------------------------------
# Translate existing arrays
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef translate_int64(carray input_, carray output_, dict lookup, npy_int64 default=-1):
    cdef:
        chunk chunk_
        Py_ssize_t i, chunklen, leftover_elements
        ndarray[npy_int64] in_buffer
        ndarray[npy_int64] out_buffer

    chunklen = input_.chunklen
    out_buffer = np.empty(chunklen, dtype='int64')
    in_buffer = np.empty(chunklen, dtype='int64')

    for i in range(input_.nchunks):
        chunk_ = input_.chunks[i]
        # decompress into in_buffer
        chunk_._getitem(0, chunklen, in_buffer.data)
        for i in range(chunklen):
            element = in_buffer[i]
            out_buffer[i] = lookup.get(element, default)
        # compress out_buffer into labels
        output_.append(out_buffer.astype(np.int64))

    leftover_elements = cython.cdiv(input_.leftover, input_.atomsize)
    if leftover_elements > 0:
        in_buffer = input_.leftover_array
        for i in range(leftover_elements):
            element = in_buffer[i]
            out_buffer[i] = lookup.get(element, default)
        output_.append(out_buffer[:leftover_elements].astype(np.int64))

# ---------------------------------------------------------------------------
# Aggregation Section (old)
@cython.boundscheck(False)
@cython.wraparound(False)
def agg_sum_na(iter_):
    cdef:
        npy_float64 v, v_cum = 0.0

    for v in iter_:
        if v == v:  # skip NA values
            v_cum += v

    return v_cum

@cython.boundscheck(False)
@cython.wraparound(False)
def agg_sum(iter_):
    cdef:
        npy_float64 v, v_cum = 0.0

    for v in iter_:
        v_cum += v

    return v_cum

# ---------------------------------------------------------------------------
# Aggregation Section
@cython.boundscheck(False)
@cython.wraparound(False)
def groupsort_indexer(carray index, Py_ssize_t ngroups):
    cdef:
        Py_ssize_t i, label, n
        ndarray[int64_t] counts, where, np_result
        # --
        carray c_result
        chunk input_chunk, index_chunk
        Py_ssize_t index_chunk_nr, index_chunk_len, leftover_elements

        ndarray[int64_t] in_buffer

    index_chunk_len = index.chunklen
    in_buffer = np.empty(index_chunk_len, dtype='int64')
    index_chunk_nr = 0

    # count group sizes, location 0 for NA
    counts = np.zeros(ngroups + 1, dtype=np.int64)
    n = len(index)

    for index_chunk_nr in range(index.nchunks):
        # fill input buffer
        input_chunk = index.chunks[index_chunk_nr]
        input_chunk._getitem(0, index_chunk_len, in_buffer.data)

        # loop through rows
        for i in range(index_chunk_len):
            counts[index[i] + 1] += 1

    leftover_elements = cython.cdiv(index.leftover, index.atomsize)
    if leftover_elements > 0:
        # fill input buffer
        in_buffer = index.leftover_array

        # loop through rows
        for i in range(leftover_elements):
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

{% for count_unique_type in count_unique_types -%}
cdef count_unique_{{ count_unique_type }}(ndarray[{{ count_unique_type }}_t] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx
        int ret = 0
        {{ count_unique_type }}_t val
        khiter_t k
        npy_uint64 count = 0
{%- if count_unique_type == "float64" %}
        bint seen_na = 0
{%- endif %}
        kh_{{ count_unique_type }}_t *table

    table = kh_init_{{ count_unique_type }}()

    for i in range(n):
        val = values[i]

        if val == val:
            k = kh_get_{{ count_unique_type }}(table, val)
            if k == table.n_buckets:
                k = kh_put_{{ count_unique_type }}(table, val, &ret)
                count += 1
{%- if count_unique_type == "float64" %}
        elif not seen_na:
            seen_na = 1
            count += 1
{%- endif %}

    kh_destroy_{{ count_unique_type }}(table)

    return count

{% endfor -%}

{% for sum_type in sum_types -%}
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef agg_{{ sum_type }}(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key, agg_method=_SUM):
    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts, factor_total_chunks, leftover_elements

        ndarray[npy_{{ sum_type }}] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_{{ sum_type }}] out_buffer

        ndarray[npy_{{ sum_type }}] last_values

        npy_{{ sum_type }} v
        bint count_distinct_started = 0
        carray num_uniques

    count = 0
    ret = 0
    reverse = {}

    if agg_method == _COUNT_DISTINCT:
        num_uniques = carray([], dtype='int64')
        positions, counts = groupsort_indexer(ca_factor, nr_groups)
        start_counts = 0
        end_counts = 0
        for j in range(len(counts) - 1):
            start_counts = end_counts
            end_counts = start_counts + counts[j + 1]
            positions[start_counts:end_counts]
            num_uniques.append(
                count_unique_{{ sum_type }}(ca_input[positions[start_counts:end_counts]])
            )

        return num_uniques

    input_chunk_len = ca_input.chunklen
    in_buffer = np.empty(input_chunk_len, dtype='{{ sum_type }}')
    factor_chunk_len = ca_factor.chunklen
    factor_total_chunks = ca_factor.nchunks
    factor_chunk_nr = 0
    factor_buffer = np.empty(factor_chunk_len, dtype='int64')
    if factor_total_chunks > 0:
        factor_chunk = ca_factor.chunks[factor_chunk_nr]
        factor_chunk._getitem(0, factor_chunk_len, factor_buffer.data)
    else:
        factor_buffer = ca_factor.leftover_array
    factor_chunk_row = 0
    # if we're calculating a mean, maintain a buffer tracking the
    # the size of each group
    if agg_method == _MEAN:
        out_buffer = np.zeros(nr_groups, dtype='float64')
        count_buffer = np.zeros(nr_groups, dtype='int64')
    else:
        out_buffer = np.zeros(nr_groups, dtype='{{ sum_type }}')

    for input_chunk_nr in range(ca_input.nchunks):
        # fill input buffer
        input_chunk = ca_input.chunks[input_chunk_nr]
        input_chunk._getitem(0, input_chunk_len, in_buffer.data)

        # loop through rows
        for i in range(input_chunk_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_chunk_len:
                factor_chunk_nr += 1
                if factor_chunk_nr < factor_total_chunks:
                    factor_chunk = ca_factor.chunks[factor_chunk_nr]
                    factor_chunk._getitem(0, factor_chunk_len, factor_buffer.data)
                else:
                    factor_buffer = ca_factor.leftover_array
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                if agg_method == _SUM:
                    out_buffer[current_index] += in_buffer[i]
                elif agg_method == _MEAN:
                    # method from Knuth
                    count_buffer[current_index] += 1
                    delta = in_buffer[i] - out_buffer[current_index]
                    out_buffer[current_index] += delta / count_buffer[current_index]
                elif agg_method == _COUNT:
                    out_buffer[current_index] += 1
                elif agg_method == _COUNT_NA:
{% if sum_type == "float64" %}
                    v = in_buffer[i]
                    if v == v:  # skip NA values
                        out_buffer[current_index] += 1
{%- else %}
                    # TODO: Warning: int does not support NA values, is this what we need?
                    out_buffer[current_index] += 1
{%- endif %}
                elif agg_method == _SORTED_COUNT_DISTINCT:
                    v = in_buffer[i]
                    if not count_distinct_started:
                        count_distinct_started = 1
                        last_values = np.zeros(nr_groups, dtype='{{ sum_type }}')
                        last_values[0] = v
                        out_buffer[0] = 1
                    else:
                        if v != last_values[current_index]:
                            out_buffer[current_index] += 1

                    last_values[current_index] = v
                else:
                    raise NotImplementedError('sumtype not supported')

    leftover_elements = cython.cdiv(ca_input.leftover, ca_input.atomsize)
    if leftover_elements > 0:
        # fill input buffer
        in_buffer = ca_input.leftover_array

        # loop through rows
        for i in range(leftover_elements):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_chunk_len:
                factor_chunk_nr += 1
                if factor_chunk_nr < factor_total_chunks:
                    factor_chunk = ca_factor.chunks[factor_chunk_nr]
                    factor_chunk._getitem(0, factor_chunk_len, factor_buffer.data)
                else:
                    factor_buffer = ca_factor.leftover_array
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                if agg_method == _SUM:
                    out_buffer[current_index] += in_buffer[i]
                elif agg_method == _MEAN:
                    # method from Knuth
                    count_buffer[current_index] += 1
                    delta = in_buffer[i] - out_buffer[current_index]
                    out_buffer[current_index] += delta / count_buffer[current_index]
                elif agg_method == _COUNT:
                    out_buffer[current_index] += 1
                elif agg_method == _COUNT_NA:
{%- if sum_type == "float64" %}
                    v = in_buffer[i]
                    if v == v:  # skip NA values
                        out_buffer[current_index] += 1
{%- else %}
                    # TODO: Warning: int does not support NA values, is this what we need?
                    out_buffer[current_index] += 1
{%- endif %}
                elif agg_method == _SORTED_COUNT_DISTINCT:
                    v = in_buffer[i]
                    if not count_distinct_started:
                        count_distinct_started = 1
                        last_values = np.zeros(nr_groups, dtype='{{ sum_type }}')
                        last_values[0] = v
                        out_buffer[0] = 1
                    else:
                        if v != last_values[current_index]:
                            out_buffer[current_index] += 1

                    last_values[current_index] = v
                else:
                    raise NotImplementedError('sumtype not supported')

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

    return out_buffer

{% endfor -%}

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef groupby_value(carray ca_input, carray ca_factor, Py_ssize_t nr_groups, Py_ssize_t skip_key):
    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks, leftover_elements

        ndarray in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray out_buffer

    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
    in_buffer = np.empty(input_chunk_len, dtype=ca_input.dtype)
    factor_chunk_len = ca_factor.chunklen
    factor_total_chunks = ca_factor.nchunks
    factor_chunk_nr = 0
    factor_buffer = np.empty(factor_chunk_len, dtype='int64')
    if factor_total_chunks > 0:
        factor_chunk = ca_factor.chunks[factor_chunk_nr]
        factor_chunk._getitem(0, factor_chunk_len, factor_buffer.data)
    else:
        factor_buffer = ca_factor.leftover_array
    factor_chunk_row = 0
    out_buffer = np.zeros(nr_groups, dtype=ca_input.dtype)

    for input_chunk_nr in range(ca_input.nchunks):

        # fill input buffer
        input_chunk = ca_input.chunks[input_chunk_nr]
        input_chunk._getitem(0, input_chunk_len, in_buffer.data)

        for i in range(input_chunk_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_chunk_len:
                factor_chunk_nr += 1
                if factor_chunk_nr < factor_total_chunks:
                    factor_chunk = ca_factor.chunks[factor_chunk_nr]
                    factor_chunk._getitem(0, factor_chunk_len, factor_buffer.data)
                else:
                    factor_buffer = ca_factor.leftover_array
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                out_buffer[current_index] = in_buffer[i]

    leftover_elements = cython.cdiv(ca_input.leftover, ca_input.atomsize)
    if leftover_elements > 0:
        in_buffer = ca_input.leftover_array

        for i in range(leftover_elements):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_chunk_len:
                factor_chunk_nr += 1
                if factor_chunk_nr < factor_total_chunks:
                    factor_chunk = ca_factor.chunks[factor_chunk_nr]
                    factor_chunk._getitem(0, factor_chunk_len, factor_buffer.data)
                else:
                    factor_buffer = ca_factor.leftover_array
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                out_buffer[current_index] = in_buffer[i]

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
        npy_int64 previous_item
        npy_int64 actual_item
        Py_ssize_t blen
        Py_ssize_t len_subgroup = 0
        Py_ssize_t max_len_subgroup
        Py_ssize_t n
        npy_bool is_in = False
        carray ret
        ndarray x, x_ones, x_zeros
        ndarray bl_basket, bl_bool_arr

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
cpdef carray_is_in(carray col, set value_set, ndarray boolarr, bint reverse):
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
