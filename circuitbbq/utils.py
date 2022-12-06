import sympy as sym

class EdgeAttributeManager:
    EC_KEY = "EC"
    EL_KEY = "EL"
    EJ_KEY = "EJ"
    CAP_KEY = "C"
    IND_KEY = "L"
    PASSIVE_PARAMETER_KEYS = (EC_KEY, EL_KEY, EJ_KEY)
    BIAS_FLUX_KEY = "ext_flux"
    BIAS_VOLTAGE_KEY = "ext_voltage"
    BIAS_CURENT = "ext_current"
    ACTIVE_PARAMETER_KEYS = (BIAS_FLUX_KEY, BIAS_VOLTAGE_KEY)

    def capacitance_to_charging_energy(self, expr):
        return 1 / (2 * sym.sympify(expr))
    
    def charging_energy_to_capacitance(self, expr):
        return self.capacitance_to_charging_energy(expr)
    
    def inductance_to_inductive_energy(self, expr):
        return 1 / sym.sympify(expr)
    
    def inductive_energy_to_inductance(self, expr):
        return 1 / sym.sympify(expr)