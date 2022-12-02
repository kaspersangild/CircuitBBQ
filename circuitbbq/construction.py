import networkx as nx
from circuitbbq.analysis import CircuitAnalyzer

EC_KEY = "EC"
EL_KEY = "EL"
EJ_KEY = "EJ"
PASSIVE_PARAMETER_KEYS = (EC_KEY, EL_KEY, EJ_KEY)
BIAS_FLUX_KEY = "bias_flux"
BIAS_VOLTAGE_KEY = "bias_voltage"
ACTIVE_PARAMETER_KEYS = (BIAS_FLUX_KEY, BIAS_VOLTAGE_KEY)

def _check_edge_params(**kwargs):
    keys = set(kwargs.keys())
    if keys.isdisjoint(PASSIVE_PARAMETER_KEYS):
        raise ValueError(
            "Must specify atleast one passive parameter. Possible choices are {}".format(
                PASSIVE_PARAMETER_KEYS
            )
        )
    all_keys = set(PASSIVE_PARAMETER_KEYS + ACTIVE_PARAMETER_KEYS)
    if not keys.issubset(all_keys):
        raise ValueError(
            "Unknown edge attribute(s): {}. Possible attributes are: {}".format(
                keys - all_keys, all_keys
            )
        )

class CircuitBuilder:
    def add_edge(self, u, v, **attr):
        """Add elements between the nodes u and v
        
        To add a capacitor add EC=<Charging energy in units of Cooper pairs, i.e. (2*e)**2 / (2*C)>
        
        To add a Josephson junction add EJ=<Josephson energy>

        To add an inductor add EL=<Inductive energy, i.e. (2*pi*Phi_0)**2 / (2 * L), where \Phi_0 is the magnetic flux quantum>

        You can also add a bias flux or voltage to between the nodes through
        
        bias_flux=<bias flux in units of (2*pi*Phi_0)>
        
        bias_voltage=<bias voltage in units of (2*pi*Phi_0) per unit time>

        Parameters
        ----------
        u : node
            starting node
        v : node
            ending node
        """
        _check_edge_params(**attr)
        self.graph.add_edge(u, v, **attr)

    def add_cycle(self, nodes_for_cycle, **attr):
        _check_edge_params(**attr)
        nx.add_cycle(self.graph, nodes_for_cycle=nodes_for_cycle, **attr)

    def add_star(self, nodes_for_star, **attr):
        _check_edge_params(**attr)
        nx.add_star(self.graph, nodes_for_star=nodes_for_star, **attr)

    def add_path(self, nodes_for_path, **attr):
        _check_edge_params(**attr)
        nx.add_path(self.graph, nodes_for_path=nodes_for_path, **attr)

    def analyzer(self, coord2nodes, nodelist=None, charging_matrix_symbol=None, xp_pairs=None):
        """Construct CircuitAnalyzer instance

        The CircuitAnalyzer class implements a bunch of methods that are useful for circuit analysis.

        Parameters
        ----------
        coord2nodes : Matrix-like
            Transformation matrix which defines the coordinates used by the analyzer. 
            Defined such that x_nodes = coord2nodes @ x_coords, where x_nodes are the node fluxes, and x_coords are the coordinate fluxes.
        nodelist : list of nodes, optional
            Defines the ordering of nodes. If not defined, networkx's default ordering is used, which is the order in which the nodes were added. 
        charging_matrix_symbol : Symbol, optional
            If defined, use this symbol as the charging matrix instead of inverting the capacitance matrix. Useful when symbolic inversion of the capacitance matrix is expensive.
        xp_pairs : Sequence of pairs of symbols, optional
            If defined, these are the symbols used for the coordinates. 

        Returns
        -------
        CircuitAnalyzer
            CircuitAnalyzer object for the defined circuit.
        """
        return CircuitAnalyzer(
            self.graph.copy(),
            coord2nodes=coord2nodes,
            nodelist=nodelist,
            charging_matrix_symbol=charging_matrix_symbol,
            xp_pairs=xp_pairs
        )

    def __init__(self, graph: nx.Graph = None):
        if graph is None:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph = graph