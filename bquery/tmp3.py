def unique(self, ndarray[object] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx, count = 0
        int ret = 0
        object val
        char *buf
        khiter_t k
        ObjectVector uniques = ObjectVector()

    for i in range(n):
        val = values[i]
        buf = util.get_c_string(val)
        k = kh_get_str(self.table, buf)
        if k == self.table.n_buckets:
            k = kh_put_str(self.table, buf, &ret)
            # print 'putting %s, %s' % (val, count)
            count += 1
            uniques.append(val)

    # return None
    return uniques.to_array()