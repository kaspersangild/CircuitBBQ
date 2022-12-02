import numpy as np
from scipy import constants


class CQEDUnits:
    def charging_energy_from_farad(self, capacitance):
        return (2 * self.e)**2 / (2 * capacitance)
        
    
    def __init__(self, time_scale=1.0, temperature_scale=1.0):
        # Constants in SI
        hbar = constants.hbar
        e = constants.elementary_charge
        k = constants.k

        # Definitions
        self.second = time_scale
        self.kelvin = temperature_scale
        self.joule = hbar**-1 / self.second
        self.coulomb = (2 * e) ** -1

        # Derivatives
        # SI
        self.ampere = self.coulomb / self.second
        self.volt = self.joule / self.coulomb
        self.farad = self.coulomb / self.volt
        self.henry = self.second**2 / self.farad
        self.weber = self.volt * self.second
        self.hertz = self.second**-1
        self.ohm = self.volt / self.ampere
        self.siemens = self.ohm**-1
        self.watt = self.joule / self.second
        
        # Natural constants
        self.hbar = hbar * self.joule * self.second
        self.h = 2 * np.pi * self.hbar
        self.e = e * self.coulomb
        self.Phi0 = self.h / (2 * self.e)
        self.Kj = 1 / self.Phi0
        self.k = k * self.joule / self.kelvin

        # Submultiples
        self.deci = 10**-1
        self.centi = 10**-2
        self.milli = 10**-3
        self.micro = 10**-6
        self.nano = 10**-9
        self.pico = 10**-12
        self.femto = 10**-15
        self.atto = 10**-18
        self.zepto = 10**-21
        self.yocto = 10**-24

        # Multiples
        self.deca = 10**1
        self.hecto = 10**2
        self.kilo = 10**3
        self.mega = 10**6
        self.giga = 10**9
        self.tera = 10**12
        self.peta = 10**15
        self.exa = 10**18
        self.zelta = 10**21
        self.yotta = 10**24
