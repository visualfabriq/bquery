def unique(self, ndarray[int64_t] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx, count = 0
        int ret = 0
        ndarray result
        int64_t val
        khiter_t k
        Int64Vector uniques = Int64Vector()

    for i in range(n):
        val = values[i]
        k = kh_get_int64(self.table, val)
        if k == self.table.n_buckets:
            k = kh_put_int64(self.table, val, &ret)
            uniques.append(val)
            count += 1

    result = uniques.to_array()

    return result
