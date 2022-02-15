# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:49:53 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u

class Nb_110_1TM:
    def __init__(self):
        
        self.Nb  = ud.Atom('Nb')
        
        self.prop = {}
        self.prop['crystal_struc']  = 'bcc'
        self.prop['c_axis']         = 4.657*u.angstrom #Straumanis, Journal of Applied Crystallography, 3(1), 1-6 (1970)
        self.prop['a_axis']         = 4.657*u.angstrom #Straumanis, Journal of Applied Crystallography, 3(1), 1-6 (1970)
        self.prop['b_axis']         = 3.300*u.angstrom #Straumanis, Journal of Applied Crystallography, 3(1), 1-6 (1970)
        self.prop['density']        = 8622.40*u.kg/(u.m**3) #Calculated from lattice constants and number of atoms per unit cell
        self.prop['deb_wal_fac']    = 0*u.m**2
        self.prop['elastic_c11']    = 246.7e9*u.kg/(u.m*u.s**2) # Carroll, Journal of Applied Physics, 36(11), 3689-3690 (1965)
        self.prop['elastic_c12']    = 133.7e9*u.kg/(u.m*u.s**2) # Carroll, Journal of Applied Physics, 36(11), 3689-3690 (1965)
        self.prop['elastic_c13']    = 133.7e9*u.kg/(u.m*u.s**2) # Carroll, Journal of Applied Physics, 36(11), 3689-3690 (1965)
        self.prop['elastic_c33']    = 246.7e9*u.kg/(u.m*u.s**2) # Carroll, Journal of Applied Physics, 36(11), 3689-3690 (1965)
        self.prop['sound_vel']      = 5.08*u.nm/u.ps #Calculated from elastic constant and density
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis']     = 7.6e-6/u.K #Roberge, J. Less-Common Met., 40(1) (1975)
        self.prop['exp_a_axis']     = 7.6e-6/u.K #Roberge, J. Less-Common Met., 40(1) (1975)
        self.prop['exp_b_axis']     = 7.6e-6/u.K #Roberge, J. Less-Common Met., 40(1) (1975)
        self.prop['lin_therm_exp']  = 18.24e-6/u.K #Calculated from elastic constants and expansion
        self.prop['Grun_c_axis']    = 1.5 #Calculated from 'lin_therm_exp' and heat capacity
        self.prop['Grun_a_axis']    = 0
        self.prop['Grun_b_axis']    = 0
        self.prop['heat_capacity']  = 270.9*u.J/(u.kg *u.K) #Haynes, CRC Handbook of Chemistry and Physics (2012)
        self.prop['therm_cond']     = 53.7*u.W/(u.m *u.K) #Ho, Journal of Physical and Chemical Reference Data, 1(2), 279-421 (1972)
        self.prop['opt_pen_depth']  = 24*u.nm #Roughly estimated from spectroscopy
        self.prop['opt_ref_index']  = 2.15+3.37j #(800nm) Adachi, Handbook On Optical Constants Of Metals, The: In Tables And Figures. World Scientific (2012)
        self.prop['opt_ref_index_per_strain'] = 0+0j
        
    def createUnitCell(self,name,caxis,prop):
        Niobium = ud.UnitCell(name, 'Niobium', caxis,**prop)
        Niobium.add_atom(self.Nb,0); Niobium.add_atom(self.Nb,0.5)
        Niobium.add_atom(self.Nb,0); Niobium.add_atom(self.Nb,0.5)
        return Niobium