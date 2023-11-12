import sympy as sym
import numpy as np
import numpy.typing as npt
import functools
import operator
from typing import Literal, Optional
from circuitbbq.bbq.bbq_strategies import (
    BBQStrategyHarmonicOscillator,
    BBQStrategyPosistionSpace,
    BBQStrategyMomentumSpace,
)
from circuitbbq.bbq.symutils import lincomb2tuple_collected
from circuitbbq.bbq.oputils import HilbertSpace, dag


class Basis:
    def __getitem__(self, key):
        if key != 0:
            raise ValueError("Indexing Error")
        return self

    def __mul__(self, other: "Basis"):
        return TensorBasis(self, other)

    def can_build(self, expr):
        if self._can_build_as_ident(expr):
            return True
        if self._can_build_custom(expr):
            return True
        if self._can_build_then_scale(expr):
            return True
        if self._can_build_then_sum(expr):
            return True
        if self._can_build_then_pow(expr):
            return True
        if self._can_build_then_matmul(expr):
            return True
        return False

    def build(self, expr: sym.Expr) -> npt.NDArray[np.complex128]:
        if self._can_build_as_ident(expr):
            out = self._build_ident()
        elif self._can_build_then_scale(expr):
            out = self._build_then_scale(expr)
        elif self._can_build_then_sum(expr):
            out = self._build_then_sum(expr)
        elif self._can_build_custom(expr):
            out = self._build_custom(expr)
        elif self._can_build_then_pow(expr):
            out = self._build_then_pow(expr)
        elif self._can_build_then_matmul(expr):
            out = self._build_then_matmul(expr)
        else:
            raise ValueError("Cannot build expression: {}".format(expr))
        return out

    def _can_build_as_ident(self, expr: sym.Expr) -> bool:
        return expr == 1

    def _can_build_then_scale(self, expr: sym.Expr) -> bool:
        coeff = sym.sympify(1)
        op = sym.sympify(1)
        if not expr.free_symbols:
            # If no free symbols, we can build as scaled ident
            return True
        if expr.func != sym.Mul:
            return False
        for arg in sym.Mul.make_args(expr):
            if arg.free_symbols:
                op = op * arg
            else:
                coeff = coeff * arg
        if (
            coeff == 1
        ):  # If we go here it was a mul of only operators and should be handled elsewhere
            return False
        return self.can_build(op)

    def _can_build_then_sum(self, expr: sym.Expr) -> bool:
        if expr.func != sym.Add:
            return False
        for arg in sym.Add.make_args(expr):
            if not self.can_build(arg):
                return False
        return True

    def _can_build_then_pow(self, expr: sym.Expr) -> bool:
        if expr.func == sym.Pow:
            return self.can_build(expr.args[0])
        return False

    def _can_build_then_matmul(self, expr: sym.Expr) -> bool:
        if expr.func == sym.Mul:
            for arg in sym.Mul.make_args(expr):
                if not self.can_build(arg):
                    return False
            return True
        return False

    def _can_build_custom(self, sanitized_expr: sym.Expr) -> bool:
        return False

    def _build_custom(self, sanitized_expr: sym.Expr) -> npt.NDArray[np.complex128]:
        raise NotImplementedError()

    def _build_ident(self) -> npt.NDArray[np.complex128]:
        return np.eye(self.dim, dtype=np.complex128)

    def _build_then_scale(self, sanitized_expr: sym.Mul) -> npt.NDArray[np.complex128]:
        coeff = sym.sympify(1)
        op = sym.sympify(1)
        for arg in sym.Mul.make_args(sanitized_expr):
            if arg.free_symbols:
                op = op * arg
            else:
                coeff = coeff * arg
        return self.build(op) * np.complex128(coeff)

    def _build_then_sum(self, sanitized_expr: sym.Add) -> npt.NDArray[np.complex128]:
        args = sym.Add.make_args(sanitized_expr)
        return functools.reduce(operator.add, (self.build(arg) for arg in args))

    def _build_then_pow(self, sanitized_expr: sym.Pow) -> npt.NDArray[np.complex128]:
        a, n = sanitized_expr.args
        n = int(n)
        return functools.reduce(operator.matmul, n * [self.build(a)])

    def _build_then_matmul(self, sanitized_expr: sym.Mul):
        args: tuple[sym.Expr, ...] = sym.Mul.make_args(sanitized_expr)
        return functools.reduce(operator.matmul, (self.build(arg) for arg in args))

    @property
    def basis_symbols(self) -> set[sym.Symbol]:
        return set([])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def tensor_dim(self) -> tuple[int, ...]:
        return (self.dim,)

    def hilbert_space(self):
        return HilbertSpace(*self.tensor_dim)

    def transform(self, v):
        return TransformedBasis(self, v)

    def eigen_basis(self, h: sym.Expr, n: Optional[int] = None):
        h = self.build(h)
        vals, vdag = np.linalg.eigh(h)
        eig_basis = self.transform(dag(vdag)[:n])
        return vals[:n], eig_basis

    def cache(self):
        return CacheBasis(self)

    def lambdify(self, args: list[sym.Symbol], expr: sym.Expr):
        expr = sym.sympify(expr)
        return LambdifiedOperator(args, expr, self)

    @property
    def domain(self):
        return (np.arange(self.dim),)

    def __init__(self, dim):
        self._dim = dim


class BBQBasis(Basis):
    @property
    def basis_symbols(self) -> set[sym.Symbol]:
        return {self.x, self.p}

    def _can_build_custom(self, sanitized_expr):
        return self._can_build_as_x_op(sanitized_expr) or self._can_build_as_p_op(
            sanitized_expr
        )

    def _can_build_as_x_op(self, expr):
        return expr.free_symbols == set([self.x])

    def _can_build_as_p_op(self, expr):
        return expr.free_symbols == set([self.p])

    def _build_custom(self, sanitized_expr):
        if self._can_build_as_x_op(sanitized_expr):
            return self._build_x_op(sanitized_expr)
        elif self._can_build_as_p_op(sanitized_expr):
            return self._build_p_op(sanitized_expr)
        raise ValueError("Cannot build expression: {}".format(sanitized_expr))

    def _build_x_op(self, expr):
        s = self.bbq_strategy
        op_fun = sym.lambdify(self.x, expr)
        arr = s.make_pot_op(op_fun)
        return arr

    def _build_p_op(self, expr):
        s = self.bbq_strategy
        op_fun = sym.lambdify(self.p, expr)
        arr = s.make_kin_op(op_fun)
        return arr

    @property
    def dim(self):
        return np.prod(self.bbq_strategy.dim)

    @property
    def domain(self):
        return self.bbq_strategy.domain

    def __init__(
        self,
        x: sym.Symbol,
        p: sym.Symbol,
        bbq_strategy: Literal["x", "p", "ho"] = "x",
        **strategy_kwargs
    ):
        """Evaluator capable of creating new operators using the bbq package.

        Parameters
        ----------
        x : Symbol
            Symbol representing x variable.
        p : Symbol
            Symbol representing p variable.
        bbq_strategy : string
            String indicating which BBQ strategy to use. Possible values are 'x', 'p' and 'ho'.
        strategy_kwargs : dict
            These parameters are forwarded to the bbq_strategy.
            The relevant parameters are xmin, xmax and dim for 'x' and 'p' options, and order, m, w, and dim for the 'ho' option.

        """
        if bbq_strategy == "x":
            self.bbq_strategy = BBQStrategyPosistionSpace(**strategy_kwargs)
        elif bbq_strategy == "p":
            self.bbq_strategy = BBQStrategyMomentumSpace(**strategy_kwargs)
        elif bbq_strategy == "ho":
            self.bbq_strategy = BBQStrategyHarmonicOscillator(**strategy_kwargs)
        else:
            raise ValueError("Invalid BBQStrategy")
        super().__init__(dim=self.bbq_strategy.dim)
        self.x = x
        self.p = p


class BasisDecorator(Basis):
    def _can_build_custom(self, sanitized_expr):
        return self.inner._can_build_custom(sanitized_expr=sanitized_expr)

    def _can_build_as_ident(self, expr):
        return self.inner._can_build_as_ident(expr)

    def _build_custom(self, sanitized_expr):
        return self.inner._build_custom(sanitized_expr=sanitized_expr)

    @property
    def basis_symbols(self) -> set[sym.Symbol]:
        return self.inner.basis_symbols

    def __init__(self, inner: Basis, dim=None):
        self.inner = inner
        if dim is None:
            super().__init__(self.inner.dim)
        else:
            super().__init__(dim)


class TransformedBasis(BasisDecorator):
    def _build_custom(self, sanitized_expr):
        a = self.inner._build_custom(sanitized_expr=sanitized_expr)
        return self.v.dot(a).dot(self.v.conj().T)

    def basis_states(self):
        return self.v.conj()

    def __init__(self, inner: Basis, v):
        self.v = v
        super().__init__(inner, dim=self.v.shape[0])


class CacheBasis(BasisDecorator):
    def _build_custom(self, sanitized_expr):
        return self._cached_build(sanitized_expr)

    def _build_ident(self):
        id_call = self._sanitize_expr(1)
        return self._cached_build(id_call)

    def __init__(self, inner: Basis, dim=None):
        self._cached_build = functools.cache(inner.build)
        super().__init__(inner, dim)


class TensorBasis(Basis):
    def __getitem__(self, key):
        N1 = len(self.basis1.tensor_dim)
        if key < N1:
            return self.basis1[key]
        else:
            return self.basis2[key - N1]

    @property
    def basis_symbols(self) -> set[sym.Symbol]:
        return self.basis1.basis_symbols.union(self.basis2.basis_symbols)

    @property
    def tensor_dim(self):
        return self.basis1.tensor_dim + self.basis2.tensor_dim

    @property
    def domain(self):
        arrs = self.basis1.domain + self.basis2.domain
        return tuple(np.meshgrid(*arrs, indexing="ij"))

    def _split_expr(self, sanitized_expr):
        expr1 = sym.sympify(1)
        expr2 = sym.sympify(1)
        for arg in sym.Mul.make_args(sanitized_expr):
            if arg.free_symbols.issubset(self.basis1.basis_symbols):
                expr1 = arg * expr1
            else:
                expr2 = arg * expr2
        return expr1, expr2

    def _can_build_custom(self, sanitized_expr):
        expr1, expr2 = self._split_expr(sanitized_expr=sanitized_expr)
        return self.basis1.can_build(expr1) and self.basis2.can_build(expr2)

    def _build_custom(self, sanitized_expr):
        expr1, expr2 = self._split_expr(sanitized_expr=sanitized_expr)
        op1 = self.basis1.build(expr1)
        op2 = self.basis2.build(expr2)
        return np.kron(op1, op2)

    def __init__(self, basis1: Basis, basis2: Basis):
        self.basis1 = basis1
        self.basis2 = basis2
        if not self.basis1.basis_symbols.isdisjoint(self.basis2.basis_symbols):
            intersect = self.basis1.basis_symbols.intersection(
                self.basis2.basis_symbols
            )
            raise ValueError(
                "Could not tensor bases since they share symbols {}".format(intersect)
            )
        super().__init__(dim=self.basis1.dim * self.basis2.dim)


class LambdifiedOperator:
    def __call__(self, *args):
        cs = self.coeff_fun(*args)
        out = sum(c * x for c, x in zip(cs, self.ops))
        return out

    def __init__(self, args, expr, basis: Basis):
        terms = lincomb2tuple_collected(expr, basis.basis_symbols)
        coeffs = []
        ops = []
        for t in terms:
            coeffs.append(t[0])
            ops.append(basis.build(t[1]))
        self.ops = tuple(ops)
        self.coeffs = tuple(coeffs)
        self.args = tuple(args)
        self.coeff_fun = sym.lambdify(args, expr=self.coeffs)
