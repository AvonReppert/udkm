# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u

class MgO_001_1TM:
    def __init__(self):
        
        self.Mg  = ud.Atom('Mg')
        self.O   = ud.Atom('O')
        
        self.prop = {}
        self.prop['crystal_struc']  = 'cubic'
        self.prop['c_axis']         = 4.210*u.angstrom #Wyckoff, Crystal Structures, Wiley, New York (1963)
        self.prop['a_axis']         = 4.210*u.angstrom #Wyckoff, Crystal Structures, Wiley, New York (1963)
        self.prop['b_axis']         = 4.210*u.angstrom #Wyckoff, Crystal Structures, Wiley, New York (1963)
        self.prop['density']        = 3580*u.kg/(u.m**3) #Calculated from lattice constants and number of atoms per unit cell
        self.prop['deb_wal_fac']    = 0*u.m**2
        self.prop['elastic_c11']    = 289.3e9*u.kg/(u.m*u.s**2) #Durand, Physical Review, 50(5), 449 (1936)
        self.prop['elastic_c12']    = 87.7e9*u.kg/(u.m*u.s**2) #Durand, Physical Review, 50(5), 449 (1936)
        self.prop['elastic_c13']    = 87.7e9*u.kg/(u.m*u.s**2) #Durand, Physical Review, 50(5), 449 (1936)
        self.prop['elastic_c33']    = 289.3e9*u.kg/(u.m*u.s**2) #Durand, Physical Review, 50(5), 449 (1936)
        self.prop['sound_vel']      = 9.580*u.nm/u.ps #Calculated from elastic constant and density
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis']     = 7.1e-6/u.K #
        self.prop['exp_a_axis']     = 6.2e-6/u.K #
        self.prop['exp_b_axis']     = 6.2e-6/u.K #
        self.prop['lin_therm_exp']  = 9.1e-6 #Calculated from elastic constants and expansion
        self.prop['Grun_c_axis']    = 0
        self.prop['Grun_a_axis']    = 0
        self.prop['Grun_b_axis']    = 0
        self.prop['heat_capacity']  = 928*u.J/(u.kg *u.K) #White, Journal of Applied Physics, 37(1), 430-432 (1966)
        self.prop['therm_cond']     = 50*u.W/(u.m *u.K) #Slifka, Journal of research of the National Institute of Standards and Technology, 103(4),357 (1998)
        self.prop['opt_pen_depth']  = np.inf*u.nm #(800nm) below band gab
        self.prop['opt_ref_index']  = 1.7276 #(800nm) Stephens, Index of refraction of magnesium oxide, J. Res. Natl. Bur. Stand. 49 249-252 (1952)
        self.prop['opt_ref_index_per_strain'] = 0+0j
        
    def createUnitCell(self,name,caxis,prop):
        MgO = ud.UnitCell(name, 'MgO', caxis,**prop)
        MgO.add_atom(self.Mg, 0); MgO.add_atom(self.O, 0.5)
        MgO.add_atom(self.Mg, 0); MgO.add_atom(self.O, 0.5)
        MgO.add_atom(self.Mg, 0); MgO.add_atom(self.O, 0.5)
        MgO.add_atom(self.Mg, 0); MgO.add_atom(self.O, 0.5)
        return MgO