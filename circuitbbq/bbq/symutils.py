import sympy as sym


def lincomb2tuple(lincomb: sym.Expr, basis_symbols: set[sym.Symbol]):
    lincomb = sym.sympify(lincomb)
    tuples = []
    for term in sym.Add.make_args(lincomb):
        op = sym.sympify("1")
        coeff = sym.sympify("1")
        for factor in sym.Mul.make_args(term):
            if factor.free_symbols.isdisjoint(basis_symbols):
                coeff *= factor
            else:
                op *= factor
        tuples.append((coeff, op))
    return tuples


def lincomb2tuple_collected(lincomb: sym.Expr, basis_symbols: set[sym.Symbol]):
    tuples = lincomb2tuple(lincomb, basis_symbols)
    new_tuples = []
    while tuples:
        op = tuples[0][1]
        v = 0
        rem_ts = []
        for t in tuples:
            c, opt = t
            if opt == op:
                v += c
                rem_ts.append(t)
        for t in rem_ts:
            tuples.remove(t)
        new_tuples.append((v, op))
    return new_tuples
