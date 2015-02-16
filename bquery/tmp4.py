def unique(self, ndarray[object] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx, count = 0
        int ret = 0
        object val
        ndarray result
        khiter_t k
        ObjectVector uniques = ObjectVector()
        bint seen_na = 0

    for i in range(n):
        val = values[i]
        hash(val)
        if not _checknan(val):
            k = kh_get_pymap(self.table, <PyObject*>val)
            if k == self.table.n_buckets:
                k = kh_put_pymap(self.table, <PyObject*>val, &ret)
                uniques.append(val)
        elif not seen_na:
            seen_na = 1
            uniques.append(ONAN)

    result = uniques.to_array()

    return result