def merge_dicts(lhs: dict, rhs: dict) -> dict:
    keys = set(lhs.keys()) | set(rhs.keys())
    d = dict()
    for k in keys:
        lk = k in lhs
        rk = k in rhs
        if lk and rk:
            lv = lhs[k]
            rv = rhs[k]
            ld = isinstance(lv, dict)
            rd = isinstance(rv, dict)
            if ld and rd:
                v = merge_dicts(lv, rv)
                d[k] = v
            else:
                d[k] = rv
        elif lk:
            d[k] = lhs[k]
        else:
            d[k] = rhs[k]
    return d
