# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:44:45 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u

class Y_0001_1TM:
    def __init__(self):
        
        self.Y  = ud.Atom('Y')

        self.prop = {}
        self.prop['c_axis']         = 6.034*u.angstrom #Spedding, Acta Crystallographica, 9(7), 559-563 (1956)
        self.prop['a_axis']         = 3.647*u.angstrom #Spedding, Acta Crystallographica, 9(7), 559-563 (1956)
        self.prop['b_axis']         = 3.647*u.angstrom*np.sin(120*np.pi/180)
        self.prop['density']        = 4247.24*u.kg/(u.m**3) #Calculated from lattice constants and number of atoms per unit cell
        self.prop['deb_wal_fac']    = 0*u.m**2
        self.prop['elastic_c11']    = 79.0e9*u.kg/(u.m*u.s**2) #Smith, Journal of Applied Physics, 31(4), 645-647 (1960)
        self.prop['elastic_c12']    = 28.7e9*u.kg/(u.m*u.s**2) #Smith, Journal of Applied Physics, 31(4), 645-647 (1960)
        self.prop['elastic_c13']    = 20.0e9*u.kg/(u.m*u.s**2) #Smith, Journal of Applied Physics, 31(4), 645-647 (1960)
        self.prop['elastic_c33']    = 77.8e9*u.kg/(u.m*u.s**2) #Smith, Journal of Applied Physics, 31(4), 645-647 (1960)
        self.prop['sound_vel']      = 4.15*u.nm/u.ps #Calculated from elastic constant and density
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis']     = 19.7e-6/u.K #Spedding, Acta Crystallographica, 9(7), 559-563 (1956)
        self.prop['exp_a_axis']     = 6.2e-6/u.K  #Spedding, Acta Crystallographica, 9(7), 559-563 (1956)
        self.prop['exp_b_axis']     = 6.2e-6/u.K  #Spedding, Acta Crystallographica, 9(7), 559-563 (1956)
        self.prop['lin_therm_exp']  = 22.9e-6 #Calculated from elastic constants and expansion
        self.prop['Grun_c_axis']    = 1.3 #Calculated from 'lin_therm_exp' and heat capacity
        self.prop['Grun_a_axis']    = 0
        self.prop['Grun_b_axis']    = 0
        self.prop['heat_capacity']  = 291.49*u.J/(u.kg *u.K) #Jennings, The Journal of Chemical Physics, 33(6), 1849-1852 (1960)
        self.prop['therm_cond']     = 24.8*u.W/(u.m *u.K) #Ho, Journal of Physical and Chemical Reference Data, 1(2), 279-421 (1972)
        self.prop['opt_pen_depth']  = 24*u.nm #Roughly estimated from spectroscopy
        self.prop['opt_ref_index']  = 2.10+2.67j #(800nm uniaxial anisotropy!) Adachi, Handbook On Optical Constants Of Metals, The: In Tables And Figures. World Scientific (2012)
        self.prop['opt_ref_index_per_strain'] = 0+0j
        
    def createUnitCell(self,name,caxis,prop):
        Yttrium = ud.UnitCell(name, 'Yttrium', caxis,**prop)
        Yttrium.add_atom(self.Y,0); Yttrium.add_atom(self.Y,0.5)
        return Yttrium