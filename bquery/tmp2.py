def unique(self, ndarray[float64_t] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx, count = 0
        int ret = 0
        float64_t val
        khiter_t k
        Float64Vector uniques = Float64Vector()
        bint seen_na = 0

    for i in range(n):
        val = values[i]

        if val == val:
            k = kh_get_float64(self.table, val)
            if k == self.table.n_buckets:
                k = kh_put_float64(self.table, val, &ret)
                uniques.append(val)
                count += 1
        elif not seen_na:
            seen_na = 1
            uniques.append(ONAN)

    return uniques.to_array()