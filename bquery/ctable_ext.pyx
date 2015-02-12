import numpy as np
import cython
from numpy cimport ndarray, dtype, npy_intp, npy_int32, npy_uint64, npy_int64, npy_float64

from libc.stdlib cimport malloc

from libc.string cimport strcpy
from khash cimport *
from bcolz.carray_ext cimport carray, chunk

cdef enum:
    SUM_DEF = 0
    SUM_COUNT = 1
    SUM_COUNT_NA = 2
    SUM_SORTED_COUNT_DISTINCT = 3


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

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_int64_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       ndarray[npy_int64] in_buffer,
                       ndarray[npy_uint64] out_buffer,
                       kh_int64_t *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):
    cdef:
        Py_ssize_t i, idx
        int ret
        npy_int64 element
        khiter_t k

    ret = 0

    for i in range(iter_range):
        element = in_buffer[i]
        k = kh_get_int64(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            k = kh_put_int64(table, element, &ret)
            table.vals[k] = idx = count[0]
            reverse[count[0]] = element
            count[0] += 1
        out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_int64(carray carray_, carray labels=None):
    cdef:
        chunk chunk_
        Py_ssize_t n, i, count, chunklen, leftover_elements
        dict reverse
        ndarray[npy_int64] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_int64_t *table

    count = 0
    ret = 0
    reverse = {}

    n = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=n)
    out_buffer = np.empty(chunklen, dtype='uint64')
    in_buffer = np.empty(chunklen, dtype='int64')
    table = kh_init_int64()

    for i in range(carray_.nchunks):
        chunk_ = carray_.chunks[i]
        # decompress into in_buffer
        chunk_._getitem(0, chunklen, in_buffer.data)
        _factorize_int64_helper(chunklen,
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
        _factorize_int64_helper(leftover_elements,
                          carray_.dtype.itemsize + 1,
                          carray_.leftover_array,
                          out_buffer,
                          table,
                          &count,
                          reverse,
                          )

    # compress out_buffer into labels
    labels.append(out_buffer[:leftover_elements].astype(np.int64))

    kh_destroy_int64(table)

    return labels, reverse

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_int32_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       ndarray[npy_int32] in_buffer,
                       ndarray[npy_uint64] out_buffer,
                       kh_int32_t *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):
    cdef:
        Py_ssize_t i, idx
        int ret
        npy_int32 element
        khiter_t k

    ret = 0

    for i in range(iter_range):
        element = in_buffer[i]
        k = kh_get_int32(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            k = kh_put_int32(table, element, &ret)
            table.vals[k] = idx = count[0]
            reverse[count[0]] = element
            count[0] += 1
        out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_int32(carray carray_, carray labels=None):
    cdef:
        chunk chunk_
        Py_ssize_t n, i, count, chunklen, leftover_elements
        dict reverse
        ndarray[npy_int32] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_int32_t *table

    count = 0
    ret = 0
    reverse = {}

    n = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=n)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    in_buffer = np.empty(chunklen, dtype='int32')
    table = kh_init_int32()

    for i in range(carray_.nchunks):
        chunk_ = carray_.chunks[i]
        # decompress into in_buffer
        chunk_._getitem(0, chunklen, in_buffer.data)
        _factorize_int32_helper(chunklen,
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
        _factorize_int32_helper(leftover_elements,
                          carray_.dtype.itemsize + 1,
                          carray_.leftover_array,
                          out_buffer,
                          table,
                          &count,
                          reverse,
                          )

    # compress out_buffer into labels
    labels.append(out_buffer[:leftover_elements].astype(np.int64))

    kh_destroy_int32(table)

    return labels, reverse

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_float64_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       ndarray[npy_float64] in_buffer,
                       ndarray[npy_uint64] out_buffer,
                       kh_float64_t *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):
    cdef:
        Py_ssize_t i, idx
        int ret
        npy_float64 element
        khiter_t k

    ret = 0

    for i in range(iter_range):
        # TODO: Consider indexing directly into the array for efficiency
        element = in_buffer[i]
        k = kh_get_float64(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            k = kh_put_float64(table, element, &ret)
            table.vals[k] = idx = count[0]
            reverse[count[0]] = element
            count[0] += 1
        out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_float64(carray carray_, carray labels=None):
    cdef:
        chunk chunk_
        Py_ssize_t n, i, count, chunklen, leftover_elements
        dict reverse
        ndarray[npy_float64] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_float64_t *table

    count = 0
    ret = 0
    reverse = {}

    n = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=n)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    in_buffer = np.empty(chunklen, dtype='float64')
    table = kh_init_float64()

    for i in range(carray_.nchunks):
        chunk_ = carray_.chunks[i]
        # decompress into in_buffer
        chunk_._getitem(0, chunklen, in_buffer.data)
        _factorize_float64_helper(chunklen,
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
        _factorize_float64_helper(leftover_elements,
                          carray_.dtype.itemsize + 1,
                          carray_.leftover_array,
                          out_buffer,
                          table,
                          &count,
                          reverse,
                          )

    # compress out_buffer into labels
    labels.append(out_buffer[:leftover_elements].astype(np.int64))

    kh_destroy_float64(table)

    return labels, reverse

def factorize(carray carray_, carray labels=None):
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
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline sum_float64_helper(ndarray[npy_float64] out_buffer,
                       Py_ssize_t current_index,
                       ndarray[npy_float64] in_buffer,
                       Py_ssize_t i,
                       sum_type):
    if sum_type == SUM_DEF:
        out_buffer[current_index] += in_buffer[i]
    elif sum_type == SUM_COUNT:
        raise NotImplementedError('SUM_COUNT')
    elif sum_type == SUM_COUNT_NA:
        raise NotImplementedError('SUM_COUNT_NA')
    elif sum_type == SUM_SORTED_COUNT_DISTINCT:
        raise NotImplementedError('SUM_SORTED_COUNT_DISTINCT')

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline sum_int64_helper(ndarray[npy_int64] out_buffer,
                       Py_ssize_t current_index,
                       ndarray[npy_int64] in_buffer,
                       Py_ssize_t i,
                       sum_type):
    if sum_type == SUM_DEF:
        out_buffer[current_index] += in_buffer[i]
    elif sum_type == SUM_COUNT:
        raise NotImplementedError('SUM_COUNT')
    elif sum_type == SUM_COUNT_NA:
        raise NotImplementedError('SUM_COUNT_NA')
    elif sum_type == SUM_SORTED_COUNT_DISTINCT:
        raise NotImplementedError('SUM_SORTED_COUNT_DISTINCT')

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline sum_int32_helper(ndarray[npy_int32] out_buffer,
                       Py_ssize_t current_index,
                       ndarray[npy_int32] in_buffer,
                       Py_ssize_t i,
                       sum_type):
    if sum_type == SUM_DEF:
        out_buffer[current_index] += in_buffer[i]
    elif sum_type == SUM_COUNT:
        raise NotImplementedError('SUM_COUNT')
    elif sum_type == SUM_COUNT_NA:
        raise NotImplementedError('SUM_COUNT_NA')
    elif sum_type == SUM_SORTED_COUNT_DISTINCT:
        raise NotImplementedError('SUM_SORTED_COUNT_DISTINCT')

@cython.wraparound(False)
@cython.boundscheck(False)
cdef sum_float64(carray ca_input, carray ca_factor,
                 Py_ssize_t nr_groups, Py_ssize_t skip_key, sum_type=SUM_DEF):
    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks, leftover_elements

        ndarray[npy_float64] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_float64] out_buffer

    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
    in_buffer = np.empty(input_chunk_len, dtype='float64')
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
    out_buffer = np.zeros(nr_groups, dtype='float64')

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
                sum_float64_helper(out_buffer, current_index,
                                   in_buffer, i, sum_type)

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
                sum_float64_helper(out_buffer, current_index,
                                   in_buffer, i, sum_type)

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

    return out_buffer

@cython.wraparound(False)
@cython.boundscheck(False)
cdef sum_int32(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key, sum_type=SUM_DEF):
    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks, leftover_elements

        ndarray[npy_int32] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_int32] out_buffer

    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
    in_buffer = np.empty(input_chunk_len, dtype='int32')
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
    out_buffer = np.zeros(nr_groups, dtype='int32')

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
                sum_int32_helper(out_buffer, current_index,
                                 in_buffer, i, sum_type)

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
                sum_int32_helper(out_buffer, current_index,
                                 in_buffer, i, sum_type)

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

    return out_buffer

@cython.wraparound(False)
@cython.boundscheck(False)
cdef sum_int64(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key, sum_type=SUM_DEF):
    cdef:
        chunk input_chunk, factor_chunk
        Py_ssize_t input_chunk_nr, input_chunk_len
        Py_ssize_t factor_chunk_nr, factor_chunk_len, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks, leftover_elements

        ndarray[npy_int64] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_int64] out_buffer

    count = 0
    ret = 0
    reverse = {}

    input_chunk_len = ca_input.chunklen
    in_buffer = np.empty(input_chunk_len, dtype='int64')
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
    out_buffer = np.zeros(nr_groups, dtype='int64')

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
                sum_int64_helper(out_buffer, current_index,
                                 in_buffer, i, sum_type)

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
                sum_int64_helper(out_buffer, current_index,
                                 in_buffer, i, sum_type)

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

    return out_buffer

@cython.wraparound(False)
@cython.boundscheck(False)
cdef groupby_value(carray ca_input, carray ca_factor, Py_ssize_t nr_groups, Py_ssize_t skip_key):
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

def aggregate_groups_by_iter_2(ct_input,
                        ct_agg,
                        npy_uint64 nr_groups,
                        npy_uint64 skip_key,
                        carray factor_carray,
                        groupby_cols,
                        output_agg_ops,
                        dtype_list
                        ):
    total = []

    for col in groupby_cols:
        total.append(groupby_value(ct_input[col], factor_carray, nr_groups, skip_key))

    for col, agg_op in output_agg_ops:
        # TODO: input vs output column
        col_dtype = ct_agg[col].dtype
        if col_dtype == np.float64:
            total.append(sum_float64(ct_input[col], factor_carray, nr_groups, skip_key))
        elif col_dtype == np.int64:
            total.append(sum_int64(ct_input[col], factor_carray, nr_groups, skip_key))
        elif col_dtype == np.int32:
            total.append(sum_int32(ct_input[col], factor_carray, nr_groups, skip_key))
        else:
            raise NotImplementedError(
                'Column dtype ({0}) not supported for aggregation yet '
                '(only int32, int64 & float64)'.format(str(col_dtype)))

    ct_agg.append(total)

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
