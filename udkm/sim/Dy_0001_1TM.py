# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:40:56 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u

class Dy_0001_1TM:
    def __init__(self):
        
        self.Dy  = ud.Atom('Dy')
        
        self.prop = {}
        self.prop['crystal_struc']   = 'hcp'
        self.prop['mag_order']       = '<90K FM and <180K AFM' #Koehler, Journal of Applied Physics, 36(3), 1078-1087 (1965)
        self.prop['mag_order_temp']  = 180*u.K
        self.prop['c_axis']          = 5.645*u.angstrom #Spedding, Acta Crystallographica, 9(7), 559-563 (1956)
        self.prop['a_axis']          = 3.551*u.angstrom #Spedding, Acta Crystallographica, 9(7), 559-563 (1956)
        self.prop['b_axis']          = 3.551*u.angstrom*np.sin(120*np.pi/180)
        self.prop['density']         = 8754.85*u.kg/(u.m**3) #Calculated from lattice constants and number of atoms per unit cell
        self.prop['molar_vol']       = 19.01e-6*u.m**3/u.mol #Wikipedia
        self.prop['molar_mass']      = 0.1664*u.kg/u.mol #Calculated from molar volume and density
        self.prop['deb_wal_fac']     = 0*u.m**2
        self.prop['elastic_c11']     = 74.2*u.GPa #Palmer, Proceedings of the Royal Society of London, 327(1571), 519-543 (1972)
        self.prop['elastic_c12']     = 25.5*u.GPa #Palmer, Proceedings of the Royal Society of London, 327(1571), 519-543 (1972)
        self.prop['elastic_c13']     = 22.5*u.GPa #Palmer, Proceedings of the Royal Society of London, 327(1571), 519-543 (1972)
        self.prop['elastic_c33']     = 78.3*u.GPa #Palmer, Proceedings of the Royal Society of London, 327(1571), 519-543 (1972)
        self.prop['sound_vel']       = 3.10*u.nm/u.ps #Calculated from elastic constant and density
        self.prop['phonon_damping']  = 0*u.kg/u.s
        self.prop['exp_c_axis']      = 20.3e-6/u.K #Spedding, Acta Crystallographica, 9(7), 559-563 (1956) ; Darnell, Physical Review, 130(5), 1825 (1963)
        self.prop['exp_a_axis']      = 4.7e-6/u.K  #Spedding, Acta Crystallographica, 9(7), 559-563 (1956) ; Darnell, Physical Review, 130(5), 1825 (1963)
        self.prop['exp_b_axis']      = 4.7e-6/u.K  #Spedding, Acta Crystallographica, 9(7), 559-563 (1956) ; Darnell, Physical Review, 130(5), 1825 (1963) 
        self.prop['lin_therm_exp']   = 20.7e-6/u.K #Calculated from elastic constants and expansion
        self.prop['Grun_c_axis']     = 1.1 #Calculated from 'lin_therm_exp' and heat capacity
        self.prop['Grun_a_axis']     = 0
        self.prop['Grun_b_axis']     = 0
        self.prop['Grun_mag_c_axis'] = 2.9 #Calculated from 'lin_therm_exp', 'Grun_c_axis' and 'mag_Q'
        self.prop['Grun_mag_a_axis'] = 0
        self.prop['Grun_mag_b_axis'] = 0
        self.prop['heat_capacity']   = 167.3*u.J/(u.kg *u.K) #Pecharsky, Scripta materialia, 35(7), 843-848 (1996)
        self.prop['therm_cond']      = 11.7*u.W/(u.m *u.K) #Ho, Journal of Physical and Chemical Reference Data, 1(2), 279-421 (1972)
        self.prop['opt_pen_depth']   = 24*u.nm #Roughly estimated from spectroscopy
        self.prop['opt_ref_index']   = 2.68+3.21j #(800nm uniaxial anisotropy!) Adachi, Handbook On Optical Constants Of Metals, The: In Tables And Figures. World Scientific (2012)
        self.prop['opt_ref_index_per_strain'] = 0+0j
        self.prop['mag_Q']           = np.genfromtxt('ReferenceData/QDy.dat',delimiter="\t",comments="#") #Extracted by Sommerfeld model and Cp of Luthetium

    def createUnitCell(self,name,caxis,prop):
        Dysprosium = ud.UnitCell(name, 'Dysprosium', caxis,**prop)
        Dysprosium.add_atom(self.Dy,0); Dysprosium.add_atom(self.Dy,0.5)
        return Dysprosium  
    
    def getQMag(self,T):
        "Returns the Magnetic Energy of Dysprosium"
        Qdeposited = np.interp(T,self.prop['mag_Q'][:,0],self.prop['mag_Q'][:,3]*u.J/u.mol)
        if T <= self.prop['mag_order_temp'].magnitude:
            Qremaining = np.interp(4000,self.prop['mag_Q'][:,0],self.prop['mag_Q'][:,3]*u.J/u.mol) - Qdeposited
        else:
            Qremaining = 0
        return Qremaining