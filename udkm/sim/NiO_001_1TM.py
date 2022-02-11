# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u

class NiO_001_1TM:
    def __init__(self):
        
        self.Ni  = ud.Atom('Ni')
        self.O   = ud.Atom('O')
        
        self.prop = {}
        self.prop['crystal_struc']  = 'cubic'
        self.prop['c_axis']         = 4.175*u.angstrom #Wyckoff, Crystal Structures, Wiley, New York (1963)
        self.prop['a_axis']         = 4.175*u.angstrom #Wyckoff, Crystal Structures, Wiley, New York (1963)
        self.prop['b_axis']         = 4.175*u.angstrom #Wyckoff, Crystal Structures, Wiley, New York (1963)
        self.prop['density']        = 6790*u.kg/(u.m**3) #Uchida, The Journal of the Acoustical Society of America 51, 1602 (1972); doi: 10.1121/1.1913005
        self.prop['deb_wal_fac']    = 0*u.m**2
        self.prop['elastic_c11']    = 270e9*u.kg/(u.m*u.s**2) #Uchida, The Journal of the Acoustical Society of America 51, 1602 (1972); doi: 10.1121/1.1913005
        self.prop['elastic_c12']    = 125e9*u.kg/(u.m*u.s**2) #Uchida, The Journal of the Acoustical Society of America 51, 1602 (1972); doi: 10.1121/1.1913005
        self.prop['elastic_c13']    = 125e9*u.kg/(u.m*u.s**2) #Uchida, The Journal of the Acoustical Society of America 51, 1602 (1972); doi: 10.1121/1.1913005
        self.prop['elastic_c33']    = 270e9*u.kg/(u.m*u.s**2) #Uchida, The Journal of the Acoustical Society of America 51, 1602 (1972); doi: 10.1121/1.1913005
        self.prop['sound_vel']      = 6.305*u.nm/u.ps #Calculated from elastic constant and density
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis']     = 7.32e-6/u.K #(325K + T-dependent) Watanabe, Thermochimica Acta 218, 365-372 (1993) 
        self.prop['exp_a_axis']     = 7.32e-6/u.K #(325K + T-dependent) Watanabe, Thermochimica Acta 218, 365-372 (1993) 
        self.prop['exp_b_axis']     = 7.32e-6/u.K #(325K + T-dependent) Watanabe, Thermochimica Acta 218, 365-372 (1993) 
        self.prop['lin_therm_exp']  = 14.1e-6 #Calculated from elastic constants and expansion
        self.prop['Grun_c_axis']    = 0
        self.prop['Grun_a_axis']    = 0
        self.prop['Grun_b_axis']    = 0
        self.prop['heat_capacity']  = 621*u.J/(u.kg *u.K) #(325K + T-dependent) Watanabe, Thermochimica Acta 218, 365-372 (1993) 
        self.prop['therm_cond']     = 30.9*u.W/(u.m *u.K) #(325K + T-dependent) Watanabe, Thermochimica Acta 218, 365-372 (1993) 
        self.prop['opt_pen_depth']  = np.inf*u.nm #(800nm) below band gab
        self.prop['opt_ref_index']  = 2.2 #(800nm) Franta, Applied Surface Science 244 426–430 (2005)
        self.prop['opt_ref_index_per_strain'] = 0+0j
        
    def createUnitCell(self,name,caxis,prop):
        NiO = ud.UnitCell(name, 'NiO', caxis,**prop)
        NiO.add_atom(self.Ni, 0); NiO.add_atom(self.O, 0.5)
        NiO.add_atom(self.Ni, 0); NiO.add_atom(self.O, 0.5)
        NiO.add_atom(self.Ni, 0); NiO.add_atom(self.O, 0.5)
        NiO.add_atom(self.Ni, 0); NiO.add_atom(self.O, 0.5)
        return NiO