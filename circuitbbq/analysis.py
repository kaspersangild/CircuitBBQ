import sympy as sym
import networkx as nx
from circuitbbq.construction import EC_KEY, EJ_KEY, EL_KEY, BIAS_FLUX_KEY, BIAS_VOLTAGE_KEY
from qiplib import BBQBasis, lincomb2tuple_collected

class CircuitAnalyzer:
    @property
    def ncoords(self):
        return self.coord2nodes.shape[1]
    
    def capacitance_matrix(self):
        lap_cap = self.laplacian(name=EC_KEY, invert_weights=True) / 2  # E_C = 1 / (2 * C) in units where 2*e=1
        v = self.coord2nodes
        lap_cap = v.T @ lap_cap @ v
        return lap_cap
    
    def laplacian(self, name: str, invert_weights=False):
        B = self.incidence_matrix()
        d = self.attr_vector(name, invert_weights)
        D = sym.diag(d)
        return B @ D @ B.T   
    
    def attr_vector(self, name: str, invert_weights=False):
        attrs = nx.get_edge_attributes(self.graph, name=name)
        v = []
        for e in self.graph.edges:
            if e in attrs:
                x = sym.sympify(attrs[e])
            else:
                x = sym.sympify(0)
            if invert_weights:
                if x != 0:
                    v.append(x**-1)
                else:
                    v.append(0)
            else:
                v.append(x)
        return sym.Matrix(v)

    def incidence_matrix(self):
        return sym.Matrix(
            nx.incidence_matrix(self.graph, oriented=True, nodelist=self.nodelist)
            .toarray()
            .astype(int)
        )
    
    def edge_fluxes(self):
        x_vec = sym.Matrix(self.x)
        flux_biases = self.attr_vector(name=BIAS_FLUX_KEY)
        B = self.incidence_matrix()
        return B.T @ x_vec + flux_biases

    def kinetic(self, clean_expr=True, charging_matrix=None):
        if charging_matrix is None:
            charging_matrix = self.charging_matrix()
        else:
            charging_matrix = sym.Matrix(charging_matrix)
        p = sym.Matrix(self.ps) - self.momentum_bias()
        out = (self.p.T@charging_matrix@p)[0,0] / 2
        if clean_expr:
            out = self.clean_expr(out)
        return out
    
    def momentum_bias(self):
        V = self.coord2nodes
        B = self.incidence_matrix()
        d = self.attr_vector(EC_KEY, invert_weights=True) / 2
        D = sym.diag(d)
        vg = self.attr_vector("bias_voltage")
        return V.T @ B @ D @ vg

    def hamiltonian(self) -> sym.Expr:
        return self.potential() + self.kinetic()

    def potential(self, clean_expr=True):
        out = 0
        fluxes = self.edge_fluxes()
        els = self.attr_vector(EL_KEY)
        ejs = self.attr_vector(EJ_KEY)
        for phi, el, ej in zip(fluxes, els, ejs):
            out += el * phi**2 / 2
            out -= ej * sym.cos(phi)
        if clean_expr:
            out = self.clean_expr(out)
        return out
    
    def clean_expr(self, expr):
        out = sym.expand(expr, trig=True)
        return sum(lincomb2tuple_collected(out, self.basis_symbols))

    def length_scales(self):
        msinv = self.charging_matrix().diagonal()
        u = self.taylor_expr(self.potential(), order=2)
        kinv = []
        for x in self.xs:
            c = u.coeff(x, 2) * 2
            kinv.append(c**-1)
        l = [(mi * ki)**sym.Rational(1, 4) for mi, ki in zip(msinv, kinv)]
        return tuple(l)

    def charging_matrix(self):
        return self.capacitance_matrix().pinv()

    def bbq_factory(self, coord_idx, sparse=False, atol=10**-12, bbq_strategy="x", dim=32, **strategy_kwargs):
        """Builds BBQBasis instance for specified coordinate.
        
        The BBQBasis constructs the matrix representation of symbolic expressions.
        
        Parameters
        ----------
        coord_idx : int
            Coordinate that the BBQBasis is tied to
        sparse : bool, optional
            Determines if the matrix representation is returned as a sparse array, by default False
        atol : float, optional
            Eliminates entries smaller that this value. Only relevant if sparse=True. By default 10**-12
        bbq_strategy : str, optional
            Specifies the quantization strategy. Default is position basis, i.e. bbq_strategy="x".
            The other options are "p" for momentum basis, and "ho" for harmonic oscillator basis.
        dim : int, optional
            Dimension of the basis, by default 32
        strategy_kwargs : optional
            Additional options passed to the BBQBasis. For bbq_strategy="x" and bbq_strategy="p" these are **xmin** and **xmax**. The wavefunction is extended periodically outside this interval.
            For bbq_strategy="ho", the options are **order** and **length_scale**. The during quantization, the potential is taylor expanded to **order**, by default 6. The length_scale parameter sets the characteristic length scale of the system such that x = length_scale*(a+a^\dagger)/\sqrt{2}, by default 1.

        Returns
        -------
        BBQBasis
            Quantization basis for the specified coordinate.
        """
        return BBQBasis(self.xs[coord_idx], self.ps[coord_idx], sparse=sparse, atol=atol, bbq_strategy=bbq_strategy, dim=dim, **strategy_kwargs)
    
    def xp_scaled(self):
        ls = self.length_scales()
        x_scaled = tuple(x * l for x, l in zip(self.xs, ls))
        p_scaled = tuple(p / l for p, l in zip(self.ps, ls))
        return x_scaled, p_scaled
    
    def xp_scaled_subs(self, idxs="all"):
        if idxs == "all":
            idxs = tuple(range(self.ncoords))
        else:
            idxs = tuple(idxs)
        subs = []
        xs, ps = self.xp_scaled()
        for idx in idxs:
            x = self.xs[idx]
            p = self.ps[idx]
            x_scaled = xs[idx]
            p_scaled = ps[idx]
            subs.append((x, x_scaled))
            subs.append((p, p_scaled))
        return subs
    
    def taylor_expr(self, expr, order=6, idxs="all"):
        if idxs == "all":
            idxs = tuple(range(self.ncoords))
        else:
            idxs = tuple(idxs)
        s = sym.Dummy()
        u = expr.subs([(x, s * x) for idx, x in enumerate(self.xs) if idx in idxs])
        u = sym.series(u, s, n=order+1).removeO()
        u = u.subs(s, 1)
        return u
    
    @property
    def basis_symbols(self):
        return self.xs + self.ps
        
    def __init__(
        self,
        circuit_graph: nx.MultiDiGraph,
        coord2nodes: sym.Matrix=None,
        nodelist=None,
        xp_pairs: tuple[sym.Symbol] = None,
    ):
        self.graph = circuit_graph
        if coord2nodes is None:
            coord2nodes = sym.eye(self.graph.number_of_nodes())
        self.coord2nodes = sym.Matrix(coord2nodes)
        self.nodelist = nodelist
        if xp_pairs is None:
            xs = sym.symbols("x:{}".format(self.ncoords))
            ps = sym.symbols("p:{}".format(self.ncoords))
            xp_pairs = tuple((x, p) for x, p in zip(xs, ps))
        if len(xp_pairs) != self.ncoords:
            raise ValueError("Invalid number of xp_pairs")
        xp_pairs = tuple(xp_pairs)
        self.xs = []
        self.ps = []
        for x, p in xp_pairs:
            self.xs.append(sym.sympify(x))
            self.ps.append(sym.sympify(p))
        self.xs = tuple(self.xs)
        self.ps = tuple(self.ps)
        if charging_matrix_symbol is not None:
            n = self.ncoords
            charging_matrix_symbol = sym.MatrixSymbol(charging_matrix_symbol, n, n)