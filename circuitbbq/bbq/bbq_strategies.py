import numpy as np
from circuitbbq.bbq.oputils import bosonic
from circuitbbq.bbq.taylor_expansion import taylor_coeffs
from scipy.special import binom
from scipy.interpolate import make_interp_spline


class BBQStrategy:
    def make_pot_op(self, func):
        raise NotImplementedError()

    def make_kin_op(self, func):
        raise NotImplementedError()

    @property
    def domain(self):
        raise NotImplementedError()


class BBQStrategyHarmonicOscillator(BBQStrategy):
    def _get_cn_dict(self, order):
        out: dict[tuple[int, int], int] = {(0, 0): 1}
        for N in range(1, order + 1):
            for n in range(N + 1):
                out[(N, n)] = 0
                if (N - 1, n - 1) in out:
                    out[(N, n)] += out[(N - 1, n - 1)]
                if (N - 1, n + 1) in out:
                    out[(N, n)] += out[(N - 1, n + 1)] * (n + 1)
        return out

    def get_bos_coeffs(self, order):
        """Function for rewriting x**N

        Builds a dictionary with values c_{N,n,k} = out[(N,n,k)], such that X**N = \sum_{n=0}^d\sum_{k=0}^nc_{N,n,k}a_+^k*a_-^(n-k)

        """
        cn_dict = self._get_cn_dict(order=order)
        out = dict()
        for N, n in cn_dict:
            cn = cn_dict[(N, n)]
            for k in range(n + 1):
                out[(N, n, k)] = cn * binom(n, k)
        return out

    def get_op(self, n, k):
        return bosonic(k_raise=k, k_lower=n - k, n=self.dim)

    def make_pot_op(self, func):
        coeffs = taylor_coeffs(func, ndim=1, order=self.order)

        # This need changing in order to work in multidim case.
        ord = coeffs.shape[0]
        bc = self.get_bos_coeffs(order=ord)
        d = self.dim
        r = self.ls / np.sqrt(2)
        out = np.zeros((d, d), dtype=np.complex128)
        for n in range(ord):
            for k in range(max(0, n - d + 1), min(n + 1, d)):
                op = self.get_op(n, k)
                for N, c in enumerate(coeffs):
                    if (N, n, k) in bc:
                        out += bc[(N, n, k)] * c * op * r**N
        return out

    def make_kin_op(self, func):
        coeffs = taylor_coeffs(func, ndim=1, order=self.order)

        # This need changing in order to work in multidim case.
        ord = coeffs.shape[0]
        bc = self.get_bos_coeffs(order=ord)
        d = self.dim
        r = self.ls**-1 / np.sqrt(2)
        out = np.zeros((d, d), dtype=np.complex128)
        for n in range(ord):
            for k in range(max(0, n - d + 1), min(n + 1, d)):
                op = self.get_op(n, k)
                for N, c in enumerate(coeffs):
                    if (N, n, k) in bc:
                        out += bc[(N, n, k)] * c * op * r**N * (-1j) ** n * (-1) ** k
        return out

    @property
    def domain(self):
        return (np.arange(self.dim),)

    def __init__(self, order=6, length_scale=1.0, dim=5):
        self.ls = float(length_scale)
        self.dim = np.int32(dim)
        if self.dim.size > 1:
            raise NotImplementedError(
                "Strategy not implemented for multidimensional case."
            )
        self.order = order


class BBQStrategyPosistionSpace(BBQStrategy):
    def make_pot_op(self, func):
        op = np.diag(func(*self.x)).astype(np.complex128)
        return op

    def make_kin_op(self, func):
        ket_axes = np.arange(len(self.dim))
        bra_axes = ket_axes + len(self.dim)
        op = np.diag(func(*self.p)).astype(np.complex128)
        op = np.fft.ifftshift(op.reshape(2 * list(self.dim)))
        op = np.fft.fftn(op, axes=bra_axes)
        op = np.fft.ifftn(op, axes=ket_axes)
        return op.reshape(np.prod(self.dim), -1)

    @property
    def domain(self):
        return self.x

    def __init__(self, xmin=-np.pi, xmax=np.pi, dim=32):
        xmin, xmax, dim = np.broadcast_arrays(np.atleast_1d(xmin), xmax, dim)
        self.xmin = xmin.astype(np.float64)
        self.xmax = xmax.astype(np.float64)
        self.dim = dim.astype(np.int32)
        self.d = (self.xmax - self.xmin) / self.dim
        self.n_modes = self.dim.size
        if self.dim.shape != self.xmin.shape or self.dim.shape != self.xmax.shape:
            raise ValueError("xmin, xmax and dim must have same length.")
        if len(self.dim.shape) != 1:
            raise ValueError("xmin, xmax and dim must be scalar or 1D arrays.")
        xs = []
        ps = []
        for x1, x2, n in zip(self.xmin, self.xmax, self.dim):
            L = x2 - x1
            d = L / n
            xs.append(np.linspace(x1, x2, n, endpoint=False))
            p = 2 * np.pi * np.fft.fftfreq(n, d=d)
            ps.append(np.fft.fftshift(p))
        self.x = tuple(x.reshape(-1) for x in np.meshgrid(*xs, indexing="ij"))
        self.p = tuple(p.reshape(-1) for p in np.meshgrid(*ps, indexing="ij"))


class BBQStrategyMomentumSpace(BBQStrategyPosistionSpace):
    def make_pot_op(self, func):
        ket_axes = np.arange(len(self.dim))
        bra_axes = ket_axes + len(self.dim)
        op = np.diag(func(*self.x)).astype(np.complex128).reshape(2 * list(self.dim))
        op = np.fft.ifftn(op, axes=bra_axes)
        op = np.fft.fftn(op, axes=ket_axes)
        op = np.fft.fftshift(op)
        return op.reshape(np.prod(self.dim), -1)

    def make_kin_op(self, func):
        op = np.diag(func(*self.p)).astype(np.complex128)
        return op

    @property
    def domain(self):
        return self.p
