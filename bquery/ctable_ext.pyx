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
        chunk chunk_
        Py_ssize_t n, i, count, chunklen, leftover_elements
        dict reverse
        np.ndarray in_buffer
        np.ndarray[np.uint64_t] out_buffer
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
        chunk chunk_
        Py_ssize_t n, i, count, chunklen, leftover_elements
        dict reverse
        np.ndarray[numpy_native_number_input] in_buffer
        np.ndarray[np.uint64_t] out_buffer
        void *table

    count = 0
    reverse = {}

    n = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=n)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    in_buffer = np.empty(chunklen, dtype=p_dtype)

    if numpy_native_number_input is np.int64_t:
        table = kh_init_int64()
    elif numpy_native_number_input is np.int32_t:
        table = kh_init_int32()
    elif numpy_native_number_input is np.float64_t:
        table = kh_init_float64()
    else:
        table = kh_init_int64()


    for i in range(carray_.nchunks):
        chunk_ = carray_.chunks[i]
        # decompress into in_buffer
        chunk_._getitem(0, chunklen, in_buffer.data)
        _factorize_number_helper[numpy_native_number_input](chunklen,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        <void*> table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer.astype(np.int64))

    leftover_elements = cython.cdiv(carray_.leftover, carray_.atomsize)
    if leftover_elements > 0:
        _factorize_number_helper[numpy_native_number_input](leftover_elements,
                          carray_.dtype.itemsize + 1,
                          carray_.leftover_array,
                          out_buffer,
                          table,
                          &count,
                          reverse,
                          )

    # compress out_buffer into labels
    labels.append(out_buffer[:leftover_elements].astype(np.int64))

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
        Py_ssize_t i, label, n
        np.ndarray[int64_t] counts, where, np_result
        # --
        carray c_result
        chunk input_chunk, index_chunk
        Py_ssize_t index_chunk_nr, index_chunk_len, leftover_elements

        np.ndarray[int64_t] in_buffer

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

            if not np.isnan(val):
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

            if not np.isnan(val):
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

            if not np.isnan(val):
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
cpdef aggregate_sum(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key,
               np.ndarray[numpy_native_number_input] in_buffer,
               np.ndarray[numpy_native_number_output] out_buffer):

    # fused type conversion
    if numpy_native_number_input is np.int64_t:
       p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
       p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
       p_dtype = np.float64

    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts, factor_total_chunks, leftover_elements

        np.ndarray[np.int64_t] factor_buffer

        np.ndarray[numpy_native_number_input] last_values

        numpy_native_number_input v
        bint count_distinct_started = 0
        carray num_uniques

        kh_str_t *table
        char *element_1
        char *element_2
        char *element_3
        int ret, size_1, size_2, size_3

    # for count distinct
    table = kh_init_str()
    sep = '|'.encode()
    last_values = np.zeros(nr_groups, dtype=p_dtype)

    # standard
    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
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
                out_buffer[current_index] += <numpy_native_number_output> in_buffer[i]

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
                out_buffer[current_index] += <numpy_native_number_output> in_buffer[i]

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef aggregate_mean(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key,
               np.ndarray[numpy_native_number_input] in_buffer,
               np.ndarray[np.float64_t] out_buffer):

    # fused type conversion
    if numpy_native_number_input is np.int64_t:
       p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
       p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
       p_dtype = np.float64

    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts, factor_total_chunks, leftover_elements

        np.ndarray[np.int64_t] factor_buffer

        np.ndarray[numpy_native_number_input] last_values

        numpy_native_number_input v
        bint count_distinct_started = 0
        carray num_uniques

        kh_str_t *table
        char *element_1
        char *element_2
        char *element_3
        int ret, size_1, size_2, size_3

    # for count distinct
    table = kh_init_str()
    sep = '|'.encode()
    last_values = np.zeros(nr_groups, dtype=p_dtype)

    # standard
    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
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

    # create special buffers for complex operations
    count_buffer = np.zeros(nr_groups, dtype='int64')

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
                # method from Knuth
                count_buffer[current_index] += 1
                delta = in_buffer[i] - out_buffer[current_index]
                out_buffer[current_index] += delta / count_buffer[current_index]

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
                # method from Knuth
                count_buffer[current_index] += 1
                delta = in_buffer[i] - out_buffer[current_index]
                out_buffer[current_index] += delta / count_buffer[current_index]

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef aggregate_std(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key,
               np.ndarray[numpy_native_number_input] in_buffer,
               np.ndarray[np.float64_t] out_buffer):

    # fused type conversion
    if numpy_native_number_input is np.int64_t:
       p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
       p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
       p_dtype = np.float64

    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts, factor_total_chunks, leftover_elements

        np.ndarray[np.int64_t] factor_buffer

        np.ndarray[numpy_native_number_input] last_values

        numpy_native_number_input v
        bint count_distinct_started = 0
        carray num_uniques

        kh_str_t *table
        char *element_1
        char *element_2
        char *element_3
        int ret, size_1, size_2, size_3

    # for count distinct
    table = kh_init_str()
    sep = '|'.encode()
    last_values = np.zeros(nr_groups, dtype=p_dtype)

    # standard
    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
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

    # create special buffers for complex operations
    count_buffer = np.zeros(nr_groups, dtype='int64')
    mean_buffer = np.zeros(nr_groups, dtype='float64')

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
                count_buffer[current_index] += 1
                delta = in_buffer[i] - mean_buffer[current_index]
                mean_buffer[current_index] += delta / count_buffer[current_index]
                # M2 = M2 + delta*(x - mean)
                out_buffer[current_index] += delta * (in_buffer[i] - mean_buffer[current_index])

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
                count_buffer[current_index] += 1
                delta = in_buffer[i] - mean_buffer[current_index]
                mean_buffer[current_index] += delta / count_buffer[current_index]
                # M2 = M2 + delta*(x - mean)
                out_buffer[current_index] += delta * (in_buffer[i] - mean_buffer[current_index])

    for i in range(len(out_buffer)):
        out_buffer[i] = np.sqrt(out_buffer[i] / (count_buffer[i]))

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef aggregate_count(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key,
               np.ndarray[numpy_native_number_input] in_buffer,
               np.ndarray[np.int64_t] out_buffer):

    # fused type conversion
    if numpy_native_number_input is np.int64_t:
       p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
       p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
       p_dtype = np.float64

    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts, factor_total_chunks, leftover_elements

        np.ndarray[np.int64_t] factor_buffer

        np.ndarray[numpy_native_number_input] last_values

        numpy_native_number_input v
        bint count_distinct_started = 0
        carray num_uniques

        kh_str_t *table
        char *element_1
        char *element_2
        char *element_3
        int ret, size_1, size_2, size_3

    # for count distinct
    table = kh_init_str()
    sep = '|'.encode()
    last_values = np.zeros(nr_groups, dtype=p_dtype)

    # standard
    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
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
                v = in_buffer[i]
                if not np.isnan(v):  # skip NA values
                    out_buffer[current_index] += 1

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
                v = in_buffer[i]
                if not np.isnan(v):  # skip NA values
                    out_buffer[current_index] += 1

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef aggregate_sorted_count_distinct(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key,
               np.ndarray[numpy_native_number_input] in_buffer,
               np.ndarray[np.int64_t] out_buffer):

    # fused type conversion
    if numpy_native_number_input is np.int64_t:
       p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
       p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
       p_dtype = np.float64

    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts, factor_total_chunks, leftover_elements

        np.ndarray[np.int64_t] factor_buffer

        np.ndarray[numpy_native_number_input] last_values

        numpy_native_number_input v
        bint count_distinct_started = 0
        carray num_uniques

        kh_str_t *table
        char *element_1
        char *element_2
        char *element_3
        int ret, size_1, size_2, size_3

    # for count distinct
    table = kh_init_str()
    sep = '|'.encode()
    last_values = np.zeros(nr_groups, dtype=p_dtype)

    # standard
    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
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

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef aggregate_count_distinct(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key,
               np.ndarray[numpy_native_number_input] in_buffer,
               np.ndarray[np.int64_t] out_buffer):

    # fused type conversion
    if numpy_native_number_input is np.int64_t:
       p_dtype = np.int64
    elif numpy_native_number_input is np.int32_t:
       p_dtype = np.int32
    elif numpy_native_number_input is np.float64_t:
       p_dtype = np.float64

    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts, factor_total_chunks, leftover_elements

        np.ndarray[np.int64_t] factor_buffer

        np.ndarray[numpy_native_number_input] last_values

        numpy_native_number_input v
        bint count_distinct_started = 0
        carray num_uniques

        kh_str_t *table
        char *element_1
        char *element_2
        char *element_3
        int ret, size_1, size_2, size_3

    # for count distinct
    table = kh_init_str()
    sep = '|'.encode()
    last_values = np.zeros(nr_groups, dtype=p_dtype)

    # standard
    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
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
                v = in_buffer[i]
                # index
                tmp_str = str(current_index).encode()
                size_1 = len(tmp_str) + 1
                element_1 = <char *>malloc(size_1)
                strcpy(element_1, tmp_str)
                # value
                tmp_str = str(v).encode()
                size_2 = len(tmp_str) + 1
                element_2 = <char *>malloc(size_2)
                strcpy(element_2, tmp_str)
                # combination
                size_3 = size_1 + size_2 + 2
                element_3 = <char *>malloc(size_3)
                strcpy(element_3, element_1 + sep + element_2)
                # hash check
                k = kh_get_str(table, element_3)
                if k == table.n_buckets:
                    # first save the new element
                    k = kh_put_str(table, element_3, &ret)
                    # then up the amount of values found
                    out_buffer[current_index] += 1

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
                v = in_buffer[i]
                # index
                tmp_str = str(current_index).encode()
                size_1 = len(tmp_str) + 1
                element_1 = <char *>malloc(size_1)
                strcpy(element_1, tmp_str)
                # value
                tmp_str = str(v).encode()
                size_2 = len(tmp_str) + 1
                element_2 = <char *>malloc(size_2)
                strcpy(element_2, tmp_str)
                # combination
                size_3 = size_1 + size_2 + 2
                element_3 = <char *>malloc(size_3)
                strcpy(element_3, element_1 + sep + element_2)
                # hash check
                k = kh_get_str(table, element_3)
                if k == table.n_buckets:
                    # first save the new element
                    k = kh_put_str(table, element_3, &ret)
                    # then up the amount of values found
                    out_buffer[current_index] += 1

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef groupby_value(carray ca_input, carray ca_factor, Py_ssize_t nr_groups, Py_ssize_t skip_key):
    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks, leftover_elements

        np.ndarray in_buffer
        np.ndarray[np.int64_t] factor_buffer
        np.ndarray out_buffer

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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef apply_where_terms(ctable_iter, list op_list, list value_list, carray boolarr):
    """
    Update a boolean array with checks whether the values of a column (col) are in a set (value_set)

    At the moment we assume integer input as it fits the current use cases

    :param array_list:
    :param op_list:
    :param value_list:
    :param boolarr:
    :return:
    """
    cdef:
        Py_ssize_t total_len, out_index, in_index, chunk_len, out_check_pos, in_check_pos, leftover_elements
        np.ndarray[np.int8_t] out_buffer
        set filter_set
        bint row_bool
        int filter_val, current_val, array_nr, op_id, current_chunk_nr
        tuple row

    chunk_len = boolarr.chunklen
    out_check_pos = chunk_len - 1
    out_buffer = np.empty(chunk_len, dtype=np.int8)
    out_index = 0
    filter_val = 0
    filter_set = set()

    for row in ctable_iter:
        row_bool = True

        for current_val, op_id, input_val in zip(row, op_list, value_list):

            if op_id in [3, 4]:
                filter_set = input_val
            else:
                filter_val = input_val

            # instructions sorted on frequency
            if op_id == 3:  # in
                if current_val not in filter_set:
                    row_bool = False
                    break
            elif op_id == 1:  # ==
                if current_val != filter_val:
                    row_bool = False
                    break
            elif op_id == 2:  # !=
                if current_val == filter_val:
                    row_bool = False
                    break
            elif op_id == 4:  # nin
                if current_val in filter_set:
                    row_bool = False
                    break
            elif op_id == 5:  # >
                if current_val <= filter_val:
                    row_bool = False
                    break
            elif op_id == 6:  # >=
                if current_val < filter_val:
                    row_bool = False
                    break
            elif op_id == 7:  # <
                if current_val >= filter_val:
                    row_bool = False
                    break
            elif op_id == 8:  # <=
                if current_val > filter_val:
                    row_bool = False
                    break

        # write bool result
        out_buffer[out_index] = row_bool

        # write array if we are at the end of the buffer
        if out_index == out_check_pos:
            boolarr.append(out_buffer)
            out_index = 0
        else:
            out_index += 1

    # write dangling last array if available
    if 0 < out_index < out_check_pos:
         boolarr.append(out_buffer[0:out_index])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef apply_where_terms_pure_in(ctable_iter, list value_list, carray boolarr):
    """
    Update a boolean array with checks whether the values of a column (col) are in a set (value_set)

    At the moment we assume integer input as it fits the current use cases

    :param array_list:
    :param op_list:
    :param value_list:
    :param boolarr:
    :return:
    """
    cdef:
        Py_ssize_t total_len, out_index, in_index, chunk_len, out_check_pos, in_check_pos, leftover_elements
        np.ndarray[np.int8_t] out_buffer
        set filter_set
        bint row_bool
        int filter_val, current_val, array_nr, op_id, current_chunk_nr
        tuple row

    chunk_len = boolarr.chunklen
    out_check_pos = chunk_len - 1
    out_buffer = np.empty(chunk_len, dtype=np.int8)
    out_index = 0
    filter_val = 0
    filter_set = set()

    for row in ctable_iter:
        row_bool = True

        for current_val, filter_set in zip(row, value_list):
            if current_val not in filter_set:
                row_bool = False
                break

        # write bool result
        out_buffer[out_index] = row_bool

        # write array if we are at the end of the buffer
        if out_index == out_check_pos:
            boolarr.append(out_buffer)
            out_index = 0
        else:
            out_index += 1

    # write dangling last array if available
    if 0 < out_index < out_check_pos:
         boolarr.append(out_buffer[0:out_index])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calc_int_array_hash(tuple input_tuple, int index):
    cdef np.int64_t output_hash = 0x345678
    cdef np.int64_t multiplier = 1000003
    cdef np.int64_t current_val

    for current_val in input_tuple:
        index -= 1
        output_hash ^= current_val
        output_hash *= multiplier
        multiplier += (82520 + 2 * index)

    output_hash += 97531

    return output_hash

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef create_group_index(ctable_iter, int nr_arrays, carray group_arr):
    """
    Creates a boolean array with a unique number for the combination of a number of indexes

    At the moment we assume integer input as it fits the current use cases

    :param ctable_iter:
    :param group_arr:
    :return:
    """
    cdef:
        chunk chunk_
        carray current_carray
        Py_ssize_t total_len, out_index, in_index, chunk_len, out_check_pos, in_check_pos, leftover_elements
        np.ndarray[np.int64_t] out_buffer
        np.ndarray[np.int64_t] current_buffer
        list walk_array_list, cursor_list, check_pos_list, current_chunk_list, row_value_list
        set filter_set
        bint row_bool
        int filter_val, array_nr, op_id, current_chunk_nr, i
        np.int64_t current_val, row_hash
        tuple col_tuple

    chunk_len = group_arr.chunklen
    out_check_pos = chunk_len - 1

    out_buffer = np.empty(chunk_len, dtype=np.int64)
    out_index = 0

    for col_tuple in ctable_iter:
        # calculate row index and save
        out_buffer[out_index] = calc_int_array_hash(col_tuple, nr_arrays)

        # write array if we are at the end of the buffer
        if out_index == out_check_pos:
            group_arr.append(out_buffer)
            out_index = 0
        else:
            out_index += 1

    # write dangling last array if available
    if 0 < out_index < out_check_pos:
         group_arr.append(out_buffer[0:out_index])

# ---------------------------------------------------------------------------
# Temporary Section
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
        chunk chunk_
        Py_ssize_t i, chunklen, leftover_elements
        np.ndarray[np.npy_int64] in_buffer
        np.ndarray[np.npy_int64] out_buffer

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
