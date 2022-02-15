# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:52:51 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u

class Sap_1120_1TM:
    def __init__(self):
        
        self.Al  = ud.Atom('Al')
        self.O   = ud.Atom('O')
        
        self.prop = {}
        self.prop['crystal_struc']  = 'hcp'
        self.prop['c_axis']         = 4.758*u.angstrom #Lucht, Journal of applied crystallography, 36(4), 1075-1081 (2003) ; Fiquet, Physics and Chemistry of Minerals, 27(2), 103-111 (1999)
        self.prop['a_axis']         = 8.241*u.angstrom #Lucht, Journal of applied crystallography, 36(4), 1075-1081 (2003) ; Fiquet, Physics and Chemistry of Minerals, 27(2), 103-111 (1999)
        self.prop['b_axis']         = 12.804*u.angstrom #Lucht, Journal of applied crystallography, 36(4), 1075-1081 (2003) ; Fiquet, Physics and Chemistry of Minerals, 27(2), 103-111 (1999)
        self.prop['density']        = 4006.65*u.kg/(u.m**3) #Calculated from lattice constants and number of atoms per unit cell
        self.prop['deb_wal_fac']    = 0*u.m**2
        self.prop['elastic_c11']    = 496.8e9*u.kg/(u.m*u.s**2) #Tefft, Journal of research of the National Bureau of Standards. Section A, Physics and chemistry, 70(4), 277 (1966)
        self.prop['elastic_c12']    = 163.6e9*u.kg/(u.m*u.s**2) #Tefft, Journal of research of the National Bureau of Standards. Section A, Physics and chemistry, 70(4), 277 (1966)
        self.prop['elastic_c13']    = 110.9e9*u.kg/(u.m*u.s**2) #Tefft, Journal of research of the National Bureau of Standards. Section A, Physics and chemistry, 70(4), 277 (1966)
        self.prop['elastic_c33']    = 498.1e9*u.kg/(u.m*u.s**2) #Tefft, Journal of research of the National Bureau of Standards. Section A, Physics and chemistry, 70(4), 277 (1966)
        self.prop['sound_vel']      = 11.14*u.nm/u.ps #Calculated from elastic constant and density ; Sinogeikin, Physics of the Earth and Planetary Interiors, 143, 575-586 (2004)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis']     = 7.1e-6/u.K #Lucht, Journal of applied crystallography, 36(4), 1075-1081 (2003)
        self.prop['exp_a_axis']     = 6.2e-6/u.K #Lucht, Journal of applied crystallography, 36(4), 1075-1081 (2003)
        self.prop['exp_b_axis']     = 6.2e-6/u.K #Lucht, Journal of applied crystallography, 36(4), 1075-1081 (2003) 
        self.prop['lin_therm_exp']  = 9.1e-6 #Calculated from elastic constants and expansion
        self.prop['Grun_c_axis']    = 1.7 #Calculated from 'lin_therm_exp' and heat capacity
        self.prop['Grun_a_axis']    = 0
        self.prop['Grun_b_axis']    = 0
        self.prop['heat_capacity']  = 657.2*u.J/(u.kg *u.K) #Ginnings, Journal of the American Chemical Society, 75(3), 522-527 (1953)
        self.prop['therm_cond']     = 58.5*u.W/(u.m *u.K) #Dobrovinskaya, Springer Science & Business Media (2009)
        self.prop['opt_pen_depth']  = np.inf*u.nm #(800nm) below band gab
        self.prop['opt_ref_index']  = 1.76 #(800nm) Malitson, Refraction and dispersion of synthetic sapphire. JOSA, 52(12), 1377-1379 (1962)
        self.prop['opt_ref_index_per_strain'] = 0+0j
        
    def createUnitCell(self,name,caxis,prop):
        Sapphire = ud.UnitCell(name, 'Sapphire', caxis,**prop)
        Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0); Sapphire.add_atom(self.Al, 0)
        Sapphire.add_atom(self.O, 0.153); Sapphire.add_atom(self.O, 0.153); Sapphire.add_atom(self.O, 0.153); Sapphire.add_atom(self.O, 0.153); Sapphire.add_atom(self.O, 0.153); Sapphire.add_atom(self.O, 0.153)
        Sapphire.add_atom(self.O, 0.194); Sapphire.add_atom(self.O, 0.194); Sapphire.add_atom(self.O, 0.194)
        Sapphire.add_atom(self.O, 0.306); Sapphire.add_atom(self.O, 0.306); Sapphire.add_atom(self.O, 0.306)
        Sapphire.add_atom(self.O, 0.347); Sapphire.add_atom(self.O, 0.347); Sapphire.add_atom(self.O, 0.347); Sapphire.add_atom(self.O, 0.347); Sapphire.add_atom(self.O, 0.347); Sapphire.add_atom(self.O, 0.347)
        Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5); Sapphire.add_atom(self.Al, 0.5)
        Sapphire.add_atom(self.O, 0.63); Sapphire.add_atom(self.O, 0.63); Sapphire.add_atom(self.O, 0.63); Sapphire.add_atom(self.O, 0.63); Sapphire.add_atom(self.O, 0.63); Sapphire.add_atom(self.O, 0.63)
        Sapphire.add_atom(self.O, 0.669); Sapphire.add_atom(self.O, 0.669); Sapphire.add_atom(self.O, 0.669)
        Sapphire.add_atom(self.O, 0.806); Sapphire.add_atom(self.O, 0.806); Sapphire.add_atom(self.O, 0.806)
        Sapphire.add_atom(self.O, 0.847); Sapphire.add_atom(self.O, 0.847); Sapphire.add_atom(self.O, 0.847); Sapphire.add_atom(self.O, 0.847); Sapphire.add_atom(self.O, 0.847); Sapphire.add_atom(self.O, 0.847)
        return Sapphire