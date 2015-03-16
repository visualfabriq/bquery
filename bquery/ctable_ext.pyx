cimport cython
from cython.parallel import parallel, prange, threadid
from openmp cimport *

import numpy as np
from numpy cimport (ndarray, dtype, npy_intp, npy_uint8, npy_int32, npy_uint64,
                    npy_int64, npy_float64, uint64_t)

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy, memset
from libcpp.vector cimport vector

from khash cimport *
import bcolz as bz
from bcolz.carray_ext cimport carray, chunks, chunk

include "parallel_processing.pxi"

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


# ----------------------------------------------------------------------------
#                           FACTORIZE SECTION
# ----------------------------------------------------------------------------
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_str_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       char * in_buffer_ptr,
                       uint64_t * out_buffer,
                       kh_str_t *table,
                       Py_ssize_t * count,
                       unsigned int thread_id,
                       omp_lock_t kh_locks[],
                       unsigned int num_threads,
                       ) nogil:
    cdef:
        Py_ssize_t i, idx
        unsigned int j
        khiter_t k

        char * element
        char * insert

        int ret = 0
        Py_ssize_t itemsize = allocation_size - 1

    # allocate enough memory to hold the string element, add one for the
    # null byte that marks the end of the string.
    # TODO: understand why zero-filling is necessary. Without zero-filling
    # the buffer, duplicate keys occur in the reverse dict
    element = <char *>calloc(allocation_size, sizeof(char))

    # lock ensures that table is in consistend state during access
    omp_set_lock(&kh_locks[thread_id])
    for i in xrange(iter_range):
        # appends null character to element string
        memcpy(element, in_buffer_ptr, itemsize)
        in_buffer_ptr += itemsize

        k = kh_get_str(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            omp_unset_lock(&kh_locks[thread_id])
            insert = <char *>malloc(allocation_size)
            # TODO: is strcpy really the best way to copy a string?
            strcpy(insert, element)

            # acquire locks for all threads to avoid inconsistent state
            for j in xrange(num_threads):
                omp_set_lock(&kh_locks[j])
            # check whether another thread has already added the entry by now
            k = kh_get_str(table, element)
            if k != table.n_buckets:
                idx = table.vals[k]
            # if not, add element
            else:
                k = kh_put_str(table, insert, &ret)
                table.vals[k] = idx = count[0]
                count[0] += 1
            # release all locks
            for j in xrange(num_threads):
                omp_unset_lock(&kh_locks[j])
            # acquire our own lock again, to indicate we are reading the table
            omp_set_lock(&kh_locks[thread_id])
        out_buffer[i] = idx

    omp_unset_lock(&kh_locks[thread_id])
    free(element)

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_str(carray carray_, unsigned int num_threads_requested = 0, **kwargs):
    cdef:
        Py_ssize_t i, blocklen, nblocks
        khint_t j
        carray labels
        ndarray in_buffer
        ndarray[npy_uint64] out_buffer
        unsigned int thread_id
        par_info_t par_info

        kh_str_t *table
        char * out_buffer_org_ptr
        char * in_buffer_ptr
        uint64_t * out_buffer_ptr

        Py_ssize_t count = 0
        dict reverse = {}

        bint first_thread = True
        Py_ssize_t element_allocation_size = carray_.dtype.itemsize + 1
        Py_ssize_t carray_chunklen = carray_.chunklen

    labels = create_labels_carray(carray_, **kwargs)

    out_buffer = np.empty(labels.chunklen, dtype='uint64')
    out_buffer_org_ptr = out_buffer.data

    table = kh_init_str()

    block_iterator = bz.iterblocks(carray_, blen=labels.chunklen)
    nblocks = <int>(len(carray_)/labels.chunklen+0.5)+1

    with nogil, parallel(num_threads=num_threads_requested):
        # Initialise some parallel processing stuff
        omp_start_critical()
        if first_thread == True:
            (&first_thread)[0] = False
            with gil:
                par_initialise(&par_info, carray_)
        omp_end_critical()
        
        # allocate thread-local in- and out-buffers
        with gil:
            in_buffer_ptr = <char *>par_allocate_buffer(labels.chunklen, carray_.dtype.itemsize)
            out_buffer_ptr = <uint64_t *>par_allocate_buffer(labels.chunklen, labels.dtype.itemsize)

        # factorise the chunks in parallel
        for i in prange(0, nblocks, schedule='dynamic', chunksize=1):
            thread_id = threadid()

            omp_set_lock(&par_info.chunk_lock)
            with gil:
                in_buffer = np.ascontiguousarray(block_iterator.next())
                blocklen = len(in_buffer)
                memcpy(in_buffer_ptr, <char*>in_buffer.data, in_buffer.nbytes)
            omp_unset_lock(&par_info.chunk_lock)

            _factorize_str_helper(blocklen,
                            element_allocation_size,
                            in_buffer_ptr,
                            out_buffer_ptr,
                            table,
                            &count,
                            thread_id,
                            par_info.kh_locks,
                            par_info.num_threads,
                            )

            # compress out_buffer into labels
            omp_set_lock(&par_info.out_buffer_lock)
            with gil:
                out_buffer.data = <char *>out_buffer_ptr
                par_save_block(i, labels, out_buffer, blocklen, nblocks)
            omp_unset_lock(&par_info.out_buffer_lock)

        # Clean-up thread local variables
        free(in_buffer_ptr)
        free(out_buffer_ptr)

    # Clean-up some parallel processing stuff
    par_terminate(par_info)

    # restore out_buffer data pointer to allow python to free the object's data
    out_buffer.data = out_buffer_org_ptr

    # construct python dict from vectors and free element memory
    for j in range(table.n_buckets):
        if not kh_exist_str(table, j):
            continue
        reverse[table.vals[j]] = <char*>table.keys[j]
        free(<void*>table.keys[j])
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
        # TODO: Consider indexing directly into the array for efficiency
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
        Py_ssize_t len_carray, count, chunklen, len_in_buffer
        dict reverse
        ndarray[npy_int64] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_int64_t *table

    count = 0
    ret = 0
    reverse = {}

    len_carray = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=len_carray)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    table = kh_init_int64()

    for in_buffer in bz.iterblocks(carray_):
        len_in_buffer = len(in_buffer)
        _factorize_int64_helper(len_in_buffer,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer[:len_in_buffer].astype(np.int64))

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
        # TODO: Consider indexing directly into the array for efficiency
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
        Py_ssize_t len_carray, count, chunklen, len_in_buffer
        dict reverse
        ndarray[npy_int32] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_int32_t *table

    count = 0
    ret = 0
    reverse = {}

    len_carray = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=len_carray)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    table = kh_init_int32()

    for in_buffer in bz.iterblocks(carray_):
        len_in_buffer = len(in_buffer)
        _factorize_int32_helper(len_in_buffer,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer[:len_in_buffer].astype(np.int64))

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
        Py_ssize_t len_carray, count, chunklen, len_in_buffer
        dict reverse
        ndarray[npy_float64] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_float64_t *table

    count = 0
    ret = 0
    reverse = {}

    len_carray = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=len_carray)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    table = kh_init_float64()

    for in_buffer in bz.iterblocks(carray_):
        len_in_buffer = len(in_buffer)
        _factorize_float64_helper(len_in_buffer,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer[:len_in_buffer].astype(np.int64))

    kh_destroy_float64(table)

    return labels, reverse

def factorize(carray carray_, **kwargs):
    if carray_.dtype == 'int32':
        labels, reverse = factorize_int32(carray_, **kwargs)
    elif carray_.dtype == 'int64':
        labels, reverse = factorize_int64(carray_, **kwargs)
    elif carray_.dtype == 'float64':
        labels, reverse = factorize_float64(carray_, **kwargs)
    else:
        #TODO: check that the input is a string_ dtype type
        labels, reverse = factorize_str(carray_, **kwargs)
    return labels, reverse

# ---------------------------------------------------------------------------
# Translate existing arrays
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef translate_int64(carray input_, carray output_, dict lookup, npy_int64 default=-1):
    cdef:
        Py_ssize_t chunklen, leftover_elements, len_in_buffer
        ndarray[npy_int64] in_buffer
        ndarray[npy_int64] out_buffer

    chunklen = input_.chunklen
    out_buffer = np.empty(chunklen, dtype='int64')

    for in_buffer in bz.iterblocks(input_):
        len_in_buffer = len(in_buffer)
        for i in range(len_in_buffer):
            element = in_buffer[i]
            out_buffer[i] = lookup.get(element, default)
        # compress out_buffer into labels
        output_.append(out_buffer[:len_in_buffer].astype(np.int64))

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
        Py_ssize_t i, label, n, len_in_buffer
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

    for in_buffer in bz.iterblocks(index):
        len_in_buffer = len(in_buffer)
        # loop through rows
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

cdef count_unique_float64(ndarray[float64_t] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx
        int ret = 0
        float64_t val
        khiter_t k
        npy_uint64 count = 0
        bint seen_na = 0
        kh_float64_t *table

    table = kh_init_float64()

    for i in range(n):
        val = values[i]

        if val == val:
            k = kh_get_float64(table, val)
            if k == table.n_buckets:
                k = kh_put_float64(table, val, &ret)
                count += 1
        elif not seen_na:
            seen_na = 1
            count += 1

    kh_destroy_float64(table)

    return count

cdef count_unique_int64(ndarray[int64_t] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx
        int ret = 0
        int64_t val
        khiter_t k
        npy_uint64 count = 0
        kh_int64_t *table

    table = kh_init_int64()

    for i in range(n):
        val = values[i]

        if val == val:
            k = kh_get_int64(table, val)
            if k == table.n_buckets:
                k = kh_put_int64(table, val, &ret)
                count += 1

    kh_destroy_int64(table)

    return count

cdef count_unique_int32(ndarray[int32_t] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx
        int ret = 0
        int32_t val
        khiter_t k
        npy_uint64 count = 0
        kh_int32_t *table

    table = kh_init_int32()

    for i in range(n):
        val = values[i]

        if val == val:
            k = kh_get_int32(table, val)
            if k == table.n_buckets:
                k = kh_put_int32(table, val, &ret)
                count += 1

    kh_destroy_int32(table)

    return count

@cython.wraparound(False)
@cython.boundscheck(False)
cdef sum_float64(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key, agg_method=_SUM):
    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts

        ndarray[npy_float64] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_float64] out_buffer
        ndarray[npy_float64] last_values

        npy_float64 v
        bint count_distinct_started = 0
        carray num_uniques

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)

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
                count_unique_float64(ca_input[positions[start_counts:end_counts]])
            )

        return num_uniques

    factor_chunk_nr = 0
    factor_buffer = iter_ca_factor.next()
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0
    out_buffer = np.zeros(nr_groups, dtype='float64')

    for in_buffer in bz.iterblocks(ca_input):
        in_buffer_len = len(in_buffer)

        # loop through rows
        for i in range(in_buffer_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_buffer_len:
                factor_chunk_nr += 1
                factor_buffer = iter_ca_factor.next()
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                if agg_method == _SUM:
                    out_buffer[current_index] += in_buffer[i]
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
                        last_values = np.zeros(nr_groups, dtype='float64')
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

@cython.wraparound(False)
@cython.boundscheck(False)
cdef sum_int32(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key, agg_method=_SUM):
    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts

        ndarray[npy_int32] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_int32] out_buffer
        ndarray[npy_int32] last_values

        npy_int32 v
        bint count_distinct_started = 0
        carray num_uniques

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)

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
                count_unique_int32(ca_input[positions[start_counts:end_counts]])
            )

        return num_uniques

    factor_chunk_nr = 0
    factor_buffer = iter_ca_factor.next()
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0
    out_buffer = np.zeros(nr_groups, dtype='int32')

    for in_buffer in bz.iterblocks(ca_input):
        in_buffer_len = len(in_buffer)

        # loop through rows
        for i in range(in_buffer_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_buffer_len:
                factor_chunk_nr += 1
                factor_buffer = iter_ca_factor.next()
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                if agg_method == _SUM:
                    out_buffer[current_index] += in_buffer[i]
                elif agg_method == _COUNT:
                    out_buffer[current_index] += 1
                elif agg_method == _COUNT_NA:

                    # TODO: Warning: int does not support NA values, is this what we need?
                    out_buffer[current_index] += 1
                elif agg_method == _SORTED_COUNT_DISTINCT:
                    v = in_buffer[i]
                    if not count_distinct_started:
                        count_distinct_started = 1
                        last_values = np.zeros(nr_groups, dtype='int32')
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

@cython.wraparound(False)
@cython.boundscheck(False)
cdef sum_int64(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key, agg_method=_SUM):
    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts

        ndarray[npy_int64] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_int64] out_buffer
        ndarray[npy_int64] last_values

        npy_int64 v
        bint count_distinct_started = 0
        carray num_uniques

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)

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
                count_unique_int64(ca_input[positions[start_counts:end_counts]])
            )

        return num_uniques

    factor_chunk_nr = 0
    factor_buffer = iter_ca_factor.next()
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0
    out_buffer = np.zeros(nr_groups, dtype='int64')

    for in_buffer in bz.iterblocks(ca_input):
        in_buffer_len = len(in_buffer)

        # loop through rows
        for i in range(in_buffer_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_buffer_len:
                factor_chunk_nr += 1
                factor_buffer = iter_ca_factor.next()
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                if agg_method == _SUM:
                    out_buffer[current_index] += in_buffer[i]
                elif agg_method == _COUNT:
                    out_buffer[current_index] += 1
                elif agg_method == _COUNT_NA:

                    # TODO: Warning: int does not support NA values, is this what we need?
                    out_buffer[current_index] += 1
                elif agg_method == _SORTED_COUNT_DISTINCT:
                    v = in_buffer[i]
                    if not count_distinct_started:
                        count_distinct_started = 1
                        last_values = np.zeros(nr_groups, dtype='int64')
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

@cython.wraparound(False)
@cython.boundscheck(False)
cdef groupby_value(carray ca_input, carray ca_factor, Py_ssize_t nr_groups, Py_ssize_t skip_key):
    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks

        ndarray in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray out_buffer

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)


    factor_total_chunks = ca_factor.nchunks
    factor_chunk_nr = 0
    factor_buffer = iter_ca_factor.next()
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0
    out_buffer = np.zeros(nr_groups, dtype=ca_input.dtype)

    for in_buffer in bz.iterblocks(ca_input):
        in_buffer_len = len(in_buffer)

        for i in range(in_buffer_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_buffer_len:
                factor_chunk_nr += 1
                factor_buffer = iter_ca_factor.next()
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
                        dtype_list,
                        agg_method=_SUM
                        ):
    total = []

    for col in groupby_cols:
        total.append(groupby_value(ct_input[col], factor_carray, nr_groups, skip_key))

    for col, agg_op in output_agg_ops:
        # TODO: input vs output column
        col_dtype = ct_agg[col].dtype
        if col_dtype == np.float64:
            total.append(
                sum_float64(ct_input[col], factor_carray, nr_groups, skip_key,
                            agg_method=agg_method)
            )
        elif col_dtype == np.int64:
            total.append(
                sum_int64(ct_input[col], factor_carray, nr_groups, skip_key,
                          agg_method=agg_method)
            )
        elif col_dtype == np.int32:
            total.append(
                sum_int32(ct_input[col], factor_carray, nr_groups, skip_key,
                          agg_method=agg_method)
            )
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
