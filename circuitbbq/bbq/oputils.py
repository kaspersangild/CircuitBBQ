import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import typing
import itertools
from numpy.typing import ArrayLike
from scipy.special import factorial
from scipy.stats import unitary_group

_COMPLEX_DTYPE = np.complex128
_OPERATOR_CLS = sp.csr_array
_STATE_CLS = np.array
_SPARSE_FORMAT = "csr"
VECTORIZE_ORDER = "C"  # To switch to column major simply change this to "F"
TENSORIZE_ORDER = "C"

OperatorLike = typing.Union[sp.spmatrix, ArrayLike]
StateLike = ArrayLike


def realify(psi):
    og_shape = psi.shape
    psi = psi.reshape(psi.shape[0], -1)
    idx = np.argmax(np.abs(psi), axis=0)
    angle = np.angle(psi[idx, np.arange(psi.shape[-1])])
    return np.real_if_close(psi * np.exp(-1j * angle)).reshape(og_shape)


# States
class HilbertSpace:
    def __mul__(self, other: "HilbertSpace"):
        td = self.tensor_dims + other.tensor_dims
        return HilbertSpace(*td)

    def tensorize(self, inpt, axes="all"):
        if axes == "all":
            axes = range(np.ndim(inpt))
        inpt = self.new_state(inpt)
        new_shape = []
        for idx, d in enumerate(inpt.shape):
            if d == self.dim and idx in axes:
                new_shape = new_shape + list(self.tensor_dims)
            else:
                new_shape.append(d)
        return inpt.reshape(new_shape)

    def detensorize(self, inpt, axes="all"):
        if axes == "all":
            axes = range(np.ndim(inpt))
        s = np.shape(inpt)
        ts = list(self.tensor_dims)
        final_shape = []
        next_shape_piece = []
        for idx, si in enumerate(s):
            next_shape_piece.append(si)
            if next_shape_piece == ts[: len(next_shape_piece)] and idx in axes:
                if len(next_shape_piece) == len(ts):
                    final_shape.append(self.dim)
                    next_shape_piece = []
            else:
                final_shape = final_shape + next_shape_piece
                next_shape_piece = []
        return np.reshape(inpt, newshape=final_shape)

    def new_state(self, data=None):
        if data is None:
            return np.zeros(self.dim, dtype=_COMPLEX_DTYPE)
        else:
            if np.size(data) == self.dim:
                return ket(data)
            elif np.size(data) == self.dim**2:
                return dm(data)
            else:
                raise ValueError("Data shape does not match hilbert space dim")

    def qubit_product_state(self, *args):
        if len(args) != len(self.tensor_dims):
            raise ValueError("Invalid number of args")
        kets = []
        for arg, d in zip(args, self.tensor_dims):
            arg = str(arg)
            h = HilbertSpace(d)
            if arg[0] == "x":
                psi = h.ket_x([0], [1], eig=arg[1])
            elif arg[0] == "y":
                psi = h.ket_y([0], [1], eig=arg[1])
            else:
                psi = h.ket(int(arg))
            kets.append(psi)
        return tensor(*kets)

    def ket(self, *idxs):
        if len(idxs) > 1:
            out = self.new_state().reshape(self.tensor_dims)
        else:
            out = self.new_state()
        out[idxs] = 1.0
        return ket(out.reshape(-1))

    def bra(self, *idxs):
        return bra(self.ket(*idxs))

    def dm(self, *idxs):
        return dm(self.ket(*idxs))

    def ketbra(self, ket_idxs, bra_idxs):
        return self.op_basis(ket_idxs=ket_idxs, bra_idxs=bra_idxs)

    def proj(self, state_idx):
        return self.ketbra(state_idx, state_idx)

    def op_basis(self, ket_idxs, bra_idxs):
        return self.ket(*ket_idxs) @ self.bra(*bra_idxs)

    def ket_x(self, idxs1, idxs2, eig="+"):
        if eig == "+":
            return (self.ket(*idxs1) + self.ket(*idxs2)) / np.sqrt(2)
        elif eig == "-":
            return (self.ket(*idxs1) - self.ket(*idxs2)) / np.sqrt(2)
        raise ValueError("Invalid value of eig. Must be + or -")

    def ket_y(self, idxs1, idxs2, eig="+"):
        if eig == "+":
            return (self.ket(*idxs1) + 1j * self.ket(*idxs2)) / np.sqrt(2)
        elif eig == "-":
            return (self.ket(*idxs1) - 1j * self.ket(*idxs2)) / np.sqrt(2)
        raise ValueError("Invalid value of eig. Must be + or -")

    def ket_super_position(self, *idxs):
        out = sum(self.ket(*i) for i in idxs)
        out /= np.sqrt(len(idxs))
        return out

    def expand_op(self, op, idx):
        tensor_args = []
        op = oper(op)
        for jdx, d in enumerate(self.tensor_dims):
            if jdx == idx:
                if d != op.shape[1]:
                    raise ValueError(
                        "Operator shape cannot act on specified hilbert space"
                    )
                tensor_args.append(op)
            else:
                tensor_args.append(eye(d))
        return tensor(*tensor_args)

    @property
    def tensor_dims(self):
        return self._tensor_dims

    @tensor_dims.setter
    def tensor_dims(self, dims):
        if np.prod(dims) == self.dim:
            self._tensor_dims = tuple(dims)
        else:
            raise ValueError("Incompatible tensor structure")

    def idxs_iter(self, filt=None):
        rs = [range(d) for d in self.tensor_dims]
        if filt is None:
            return itertools.product(*rs)
        return filter(filt, itertools.product(*rs))

    def ketbra_idx_iter(self, order=None, filt=None):
        if order is None:
            order = VECTORIZE_ORDER
        if order == "C":
            # bra idx changes fastest
            return (
                (k, b)
                for k, b in itertools.product(
                    self.idxs_iter(filt=filt), self.idxs_iter(filt=filt)
                )
            )
        if order == "F":
            # Ket idx changes fastest
            return (
                (k, b)
                for b, k in itertools.product(
                    self.idxs_iter(filt=filt), self.idxs_iter(filt=filt)
                )
            )
        raise ValueError("Invalid order")

    def ravel_idxs(self, multi_index, mode="raise", order=TENSORIZE_ORDER):
        return np.ravel_multi_index(
            multi_index=multi_index, dims=self.tensor_dims, mode=mode, order=order
        )

    def unravel_idxs(self, indices, order=TENSORIZE_ORDER):
        return np.unravel_index(indices=indices, shape=self.tensor_dims, order=order)

    def __init__(self, *tensor_dims):
        if len(tensor_dims) == 1:
            if hasattr(tensor_dims[0], "__len__"):
                tensor_dims = tensor_dims[0]
        self.dim = np.int_(np.prod(tensor_dims))
        self._tensor_dims = tensor_dims


# Basic manipulation
def ket2dm(psi):
    return ket(psi) * bra(psi)


def dag(state_or_oper):
    if sp.issparse(state_or_oper):
        return state_or_oper.getH().asformat(_SPARSE_FORMAT)
    else:
        return np.transpose(state_or_oper).conj()


def is_ket(state):
    if np.shape(state) == (np.size(state), 1):
        return True
    return False


def is_bra(state):
    if np.shape(state) == (1, np.size(state)):
        return True
    return False


def is_dm(state):
    s = np.shape(state)
    if len(s) == 2 and s[0] == s[1]:
        return True
    return False


def schmidt_decomposition(states, d1, d2):
    states = states.T  # First index over energies. Second over basis
    states = states.reshape(-1, d1, d2)
    # states: (energies_idx, basis_idxA, basis_idxB)
    psiA, s, psiB = np.linalg.svd(states, full_matrices=False)
    # psiA: (energy_idx, basis_idxA, schmidt_idx)
    # s: (energy_idx, schmidt_idx)
    # psiB: (energy_idx, schmidt_idx, basis_idxB) <- Needs to swap axes
    psiB = np.swapaxes(psiB, 1, 2)
    return psiA, s, psiB


def operator_schmidt_decomposition(x, d1, d2):
    x = x.T
    x = np.swapaxes(x.reshape(d1, d2, d1, d2), 1, 2).reshape(d1**2, d2**2)
    a, s, b = np.linalg.svd(x, full_matrices=False)
    b = b.T
    a = a.reshape(d1, d1, -1)
    b = b.reshape(d2, d2, -1)
    return a, s, b


def random_ket(d, random_state=None):
    us = unitary_group.rvs(d, random_state=random_state)
    return us[:, 0]


def entangling_power_mc(u, d1, d2, N=500):
    ep = []
    for _ in range(N):
        psiA = random_ket(d1)
        psiB = random_ket(d2)
        psi = u.dot(tensor(ket(psiA), ket(psiB)))
        ep.append(entanglement_entropy_linear(psi, (2, 2)))
    return np.mean(ep)


def index_of_separability(a):
    s = la.svdvals(a)
    return s[0] ** 2 / np.sum(s**2)


def operator_separability(op, d1, d2=None):
    if d2 is None:
        d2 = int(op.shape[0] / d1)
    op = op.reshape(d1, d2, d1, d2)
    op = np.swapaxes(op, 1, 2)
    op = op.reshape(d1**2, d2**2)
    return index_of_separability(op)


def coupling_strength_from_eig_states(
    energies, states, D1, D2=None, d1=None, d2=None, norm1=1, norm2=1
):
    if D2 is None:
        D2 = int(states.shape[1] / D1)
    if d1 is None:
        d1 = D1
    if d2 is None:
        d2 = D2
    energies -= np.mean(energies)
    psiA, s, psiB = schmidt_decomposition(states, D1, D2)
    ha = np.einsum("e, es, eis, ejs", energies, s, psiA, psiA.conj()) / d2
    hb = np.einsum("e, es, eis, ejs", energies, s, psiB, psiB.conj()) / d1
    trH2 = np.sum(energies**2)
    trHa2 = np.trace(ha @ ha)
    trHb2 = np.trace(hb @ hb)
    Vnorm = np.sqrt(trH2 - trHa2 * d2 - trHb2 * d1)
    g = Vnorm / (norm1 * norm2)
    return g


def expect(op, state, real_if_close=True):
    if not hasattr(op, "dot"):
        op = oper(op)
    if is_dm(state):
        rho = dm(state)
        out = np.trace(op.dot(rho))
    else:
        k = op.dot(ket(state))
        b = bra(state)
        out = np.squeeze(b.dot(k))
    if real_if_close:
        return np.real_if_close(out)
    return out


# Metrics
def fidelity(state1: StateLike, state2: StateLike):
    rho, sigma = dm(state1), dm(state2)
    sqrt_rho = la.sqrtm(rho)
    trace_arg = la.sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(trace_arg)) ** 2


def process_fidelity(sop1, sop2):
    rho = super_to_choi(sop1)
    sig = super_to_choi(sop2)
    return fidelity(rho, sig)


def purity(rho: StateLike):
    rho = dm(rho)
    rho_sq = rho @ rho
    return np.trace(rho_sq).real


def von_neuman_entropy(rho: StateLike):
    rho = dm(rho)
    vals = la.eigvalsh(rho)
    vals = vals[vals > 0]
    return np.sum(-np.log(vals) * vals)


def linear_entropy(rho: StateLike, normalized=False):
    if normalized:
        e = linear_entropy(rho, normalized=False)
        d = np.shape(rho)
        return e * d / (d - 1)
    return 1 - purity(rho)


def partial_trace(rho: StateLike, dims: typing.Sequence[int]) -> StateLike:
    """Partial trace of density matrix.

    Partial trace of density operator with underlying hilbertspace H = H_1 x H_2. H_1 is traced over.

    Parameters
    ----------
    rho : StateLike
        Density matrix to be traced over. Must have shape (d1*d2, d1*d2) or (d1, d2, d1, d2)
    dims : Sequence of ints
        dims = (d1, d2)

    Returns
    -------
    Reduced density matrix

    """
    rho_tensor = dm(rho).reshape(2 * dims, order=VECTORIZE_ORDER)
    rho_b = np.einsum("aiaj", rho_tensor)
    return rho_b


def entanglement_power(unitary):
    D = unitary.shape[0]
    d = int(np.sqrt(D))
    return entangling_power(unitary, d, d)
    # Nd = (d + 1) / d
    # t = unitary.reshape(d, d, d, d)  # i_a, i_b, j_a, j_b
    # t = np.swapaxes(t, 1, 2)  # i_a, j_a, i_b, j_b
    # s1 = la.svdvals(t.reshape(D, D))
    # t = np.swapaxes(t, 1, -1)  # i_a, j_b, i_b, j_a
    # s2 = la.svdvals(t.reshape(D, D))
    # es = 1 - D**-1
    # eu = 1 - D**-2 * np.sum(s1**4)
    # eus = 1 - D**-2 * np.sum(s2**4)
    # return (eu + eus - es) * Nd**-2


def entangling_power(u, d1, d2):
    t = u.reshape(d1, d2, d1, d2)  # i_a, i_b, j_a, j_b
    t = np.swapaxes(t, 1, 2)  # i_a, j_a, i_b, j_b
    s1 = la.svdvals(t.reshape(d1**2, d2**2))
    t = np.swapaxes(t, 1, -1)  # i_a, j_b, i_b, j_a
    s2 = la.svdvals(t.reshape(d1 * d2, d1 * d2))
    r = np.sum(s1**4)
    s = np.sum(s2**4)
    return 1 - (d1 * d2 * (d1 + d2) + s + r) / (d1 * d2 * (d1 + 1) * (d2 + 1))


def _bipartite_entanglement_measure(rho, dims, measure):
    rho_b = partial_trace(rho, dims)
    return measure(rho_b)


def entanglement_entropy(rho: StateLike, dims: tuple[int]):
    return _bipartite_entanglement_measure(rho, dims, von_neuman_entropy)


def entanglement_entropy_linear(rho: StateLike, dims: tuple[int]):
    return _bipartite_entanglement_measure(rho, dims, linear_entropy)


# Operator creation
def oper(data):
    return _OPERATOR_CLS(data, dtype=_COMPLEX_DTYPE)


def state(data):
    return _STATE_CLS(data, dtype=_COMPLEX_DTYPE)


def basis_tensor(idxs: typing.Sequence[int], dims: typing.Sequence[int]):
    out = state(np.zeros(dims))
    out[tuple(idxs)] = 1.0
    return out


def basis_vector(idxs: typing.Sequence[int], dims: typing.Sequence[int]):
    return basis_tensor(idxs, dims).reshape(-1, order=VECTORIZE_ORDER)


def ket(data):
    if is_ket(data) or np.ndim(data) == 1:
        return _STATE_CLS(data, dtype=_COMPLEX_DTYPE).reshape(
            -1, 1, order=VECTORIZE_ORDER
        )
    elif is_bra(data):
        return dag(bra(data))
    else:
        raise ValueError("Data could not be interpreted as ket or bra")


def bra(data):
    if is_bra(data):
        return _STATE_CLS(data, dtype=_COMPLEX_DTYPE)
    return dag(ket(data))


def dm(data):
    if is_ket(data) or is_bra(data) or np.ndim(data) == 1:
        return ket(data) * bra(data)  # <- Broadcasting to achieve outer product
    elif is_dm(data):
        return _STATE_CLS(data, dtype=_COMPLEX_DTYPE)
    else:
        raise ValueError("Data could not be interpreted as ket, bra or dm")


def paulix():
    return oper([[0, 1], [1, 0]])


def pauliy():
    return oper([[0, -1j], [1j, 0]])


def pauliz():
    return oper([[1, 0], [0, -1]])


def hadamard():
    return oper([[1, 1], [1, -1]]) * 2**-0.5


def eye(m, n=None, k=0):
    return _OPERATOR_CLS(sp.eye(m, n, k=k, dtype=_COMPLEX_DTYPE))


def zeros(m, n=None):
    if n is None:
        n = m
    return _OPERATOR_CLS((m, n), dtype=_COMPLEX_DTYPE)


def create(n):
    dia = np.sqrt(np.arange(1, n))
    dense = np.diag(dia, k=-1)
    return oper(dense)


def destroy(n):
    dia = np.sqrt(np.arange(1, n))
    dense = np.diag(dia, k=1)
    return oper(dense)


def bosonic(k_raise, k_lower, n):
    if k_raise > n or k_lower > n:
        return np.zeros((n, n), dtype=_COMPLEX_DTYPE)
    dk = int(abs(k_raise - k_lower))
    nfac = factorial(np.arange(n))
    dl = np.sqrt(nfac[k_lower:] / nfac[: n - k_lower])
    dr = np.sqrt(nfac[k_raise:] / nfac[: n - k_raise])
    out = np.diag(dr, k=-k_raise) @ np.diag(dl, k=k_lower)
    return out.astype(_COMPLEX_DTYPE)


def num(n):
    dia = np.arange(0, n)
    dense = np.diag(dia)
    return oper(dense)


# Functions acting on operators
def remove_small(a, atol=10**-12):
    """Removes small elements

    Removes elements where abs(a[i,j])<atol IN PLACE!!

    Parameters
    ----------
    a : Matrix

    atol : float, optional
        Remove elements smaller than this value, by default 10**-12
    """
    mask = np.abs(a.data) < atol
    a.data[mask] = 0.0
    a.eliminate_zeros()


def tensor(*ops):
    if len(ops) == 1:
        return ops[0]
    else:
        if hasattr(ops[0], "toarray"):
            a = ops[0]
            for b in ops[1:]:
                a = sp.kron(a, b, format=_SPARSE_FORMAT)
            return a
        else:
            a = ops[0]
            for b in ops[1:]:
                a = la.kron(a, b)
            return a


# Super Operators
def tensorize_axis(inpt: StateLike, axis_dims: typing.Sequence[int], axis_idx: int):
    new_shape = []
    for idx, s in enumerate(np.shape(inpt)):
        if idx == axis_idx:
            new_shape = new_shape + list(axis_dims)
        else:
            new_shape.append(s)
    return np.reshape(inpt, new_shape, order=TENSORIZE_ORDER)


def detensorize_axis(inpt: StateLike, tensor_axes_start: int, tensor_axes_end: int):
    if tensor_axes_end < tensor_axes_start:
        raise ValueError("First tensor axes index mus be lower than last")
    sl, sr = [], []
    for idx, si in enumerate(np.shape(inpt)):
        if idx < tensor_axes_start:
            sl.append(si)
        if idx > tensor_axes_end:
            sr.append(si)
    return np.reshape(inpt, newshape=sl + [-1] + sr, order=TENSORIZE_ORDER)


def vectorize(mat_or_vec, order=None):
    if order is None:
        order = VECTORIZE_ORDER
    if not hasattr(mat_or_vec, "shape"):
        mat_or_vec = np.asarray(mat_or_vec)

    if len(mat_or_vec.shape) > 2:
        raise ValueError("Input must be matrix or vector")
    return mat_or_vec.reshape(-1, 1, order=order)


def devectorize(vec, order=None):
    if order is None:
        order = VECTORIZE_ORDER
    if not hasattr(vec, "shape"):
        vec = np.asarray(vec)
    if len(vec.shape) != 1:
        if len(vec.shape) != 2 or not vec.shape[1] == 1:
            raise ValueError("Input must be 1d array or column vector")
    n = np.sqrt(vec.size).astype(np.int32)
    return vec.reshape(n, n, order=order)


def devectorize_axes(array: np.ndarray, axes: typing.Sequence[int]):
    ar = np.array(array)
    new_shape = []
    for idx, dim in enumerate(ar.shape):
        if idx in axes:
            n = np.sqrt(dim).astype(np.int32)
            new_shape.append(n)
            new_shape.append(n)
        else:
            new_shape.append(dim)
    return ar.reshape(new_shape, order=VECTORIZE_ORDER)


def matmul(a):
    a = oper(a)
    return sandwich(a, eye(*a.shape))


def rmatmul(a):
    a = oper(a)
    return sandwich(eye(*a.shape), a)


def commutator(a):
    a = oper(a)
    return matmul(a) - rmatmul(a)


def anticommutator(a):
    a = oper(a)
    return matmul(a) + rmatmul(a)


def sandwich(a, b):
    a = oper(a)
    b = oper(b)
    if VECTORIZE_ORDER == "C":  # Row major
        return tensor(a, b.T)
    elif VECTORIZE_ORDER == "F":  # Column major
        return tensor(b.T, a)
    else:
        raise ValueError("Invalid order")


def _kraus(k):
    k = oper(k)
    return sandwich(k, k.conj().T)


def kraus(*ks):
    out = 0.0
    for k in ks:
        out = out + _kraus(k)
    return out


def dissipator(l, adjoint=False):
    if adjoint:
        return kraus(dag(l)) - 0.5 * anticommutator(l.conj().T @ l)
    else:
        return kraus(l) - 0.5 * anticommutator(dag(l) @ l)


# Other reps
def super_to_choi(a):
    a_op = oper(a).toarray()
    a_op = devectorize_axes(a_op, axes=[0, 1])
    hs_choi = HilbertSpace(a_op.shape[2], a_op.shape[0])
    a_op = a_op.transpose([2, 0, 3, 1]) / hs_choi.tensor_dims[0]
    return hs_choi.detensorize(a_op)


# Rotating frame
def interaction_picture(phases: np.array):
    d = np.exp(1j * phases)
    return DiagonalLinearOperator(d)


def interaction_picture_super(phases: np.array):
    d = np.exp(1j * phases)
    D = vectorize((d[:, None] * d.conj()))
    return DiagonalLinearOperator(D)


# LinearOps
class DiagonalLinearOperator(spla.LinearOperator):
    def _matmat(self, X):
        return self._d.reshape(-1, 1) * X

    def _adjoint(self):
        return DiagonalLinearOperator(self._d.conj())

    def __init__(self, d: np.array):
        self._d = np.squeeze(d)
        if self._d.ndim != 1:
            raise ValueError("Diagonal must be 1D")
        shape = 2 * (d.size,)
        dtype = self._d.dtype
        super().__init__(dtype=dtype, shape=shape)


class COOTensor:
    @staticmethod
    def from_coo_matrix(a, dense_shape):
        values = a.data
        flat_indices = np.multiply(a.shape[1], a.row) + a.col
        indices = np.unravel_index(flat_indices, dense_shape)
        return COOTensor(indices, values, dense_shape)

    def to_coo_matrix(self, row_axes, col_axes):
        dr = tuple(self.shape[ax] for ax in row_axes)
        dc = tuple(self.shape[ax] for ax in col_axes)
        shape = (np.prod(dr), np.prod(dc))
        ri = tuple(self.indices[ax] for ax in row_axes)
        ci = tuple(self.indices[ax] for ax in col_axes)
        row = np.ravel_multi_index(ri, dims=dr)
        col = np.ravel_multi_index(ci, dims=dc)
        return sp.coo_array((self.values, (row, col)), shape=shape)

    def __init__(self, indices, values, dense_shape):
        self.values = np.array(values)
        self.indices = tuple(indices)
        self.shape = tuple(dense_shape)
        self.ndims = len(self.shape)
        for i in self.indices:
            if self.values.size != i.size:
                raise ValueError("Incompatible indices and values")
        if self.indices.shape[1] != self.ndims:
            raise ValueError("Incompatible indices and shape")
