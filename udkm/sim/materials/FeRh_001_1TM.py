# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:40:20 2021

@author: matte
"""

import udkm1Dsim as ud
u = ud.u

class FeRh_001_1TM:
    def __init__(self):
        
        self.Fe  = ud.Atom('Fe')
        self.Rh  = ud.Atom('Rh')
        
        self.prop = {}
        self.prop['crystal_struc']    = 'cubic'
        self.prop['mag_order']        = '<370K AFM and <650K FM' #
        self.prop['mag_order_temp']   = 370*u.K
        self.prop['c_axis']           = 2.998*u.angstrom #On MgO: Maat, PRB, 72, 214432 (2005)
        self.prop['a_axis']           = 2.975*u.angstrom #On MgO: Maat, PRB, 72, 214432 (2005)
        self.prop['b_axis']           = 2.975*u.angstrom #On MgO: Maat, PRB, 72, 214432 (2005)
        self.prop['density']          = 9884*u.kg/(u.m**3) #Calculated from lattice constants and number of atoms per unit cell
        self.prop['molar_vol']        = 7.723e-6*u.m**3/u.mol #Calculated from 'molar_mass' and 'density'
        self.prop['molar_mass']       = 0.0794*u.kg/u.mol #Wikipedia
        self.prop['deb_wal_fac']      = 0*u.m**2
        self.prop['elastic_c11']      = 285.5*u.GPa #Palmer, Physica Status Solid (a), 32(2), 503-508 (1975)
        self.prop['elastic_c12']      = 135.6*u.GPa #Palmer, Physica Status Solid (a), 32(2), 503-508 (1975)
        self.prop['elastic_c13']      = 135.6*u.GPa #Palmer, Physica Status Solid (a), 32(2), 503-508 (1975)
        self.prop['elastic_c33']      = 285.5*u.GPa #Palmer, Physica Status Solid (a), 32(2), 503-508 (1975)
        self.prop['sound_vel']        = 5.000*u.nm/u.ps #Calculated from elastic constant and density (5.283)
        self.prop['phonon_damping']   = 0*u.kg/u.s
        self.prop['exp_c_axis']       = 9.7e-6/u.K  #Ibarra, PRB, 50(6), 4196-4199 (1994)
        self.prop['exp_a_axis']       = 9.7e-6/u.K  #Ibarra, PRB, 50(6), 4196-4199 (1994)
        self.prop['exp_b_axis']       = 9.7e-6/u.K  #Ibarra, PRB, 50(6), 4196-4199 (1994)
        self.prop['exp_c_axis_FM']    = 6e-6/u.K    #Ibarra, PRB, 50(6), 4196-4199 (1994)
        self.prop['exp_a_axis_FM']    = 6e-6/u.K    #Ibarra, PRB, 50(6), 4196-4199 (1994)
        self.prop['exp_b_axis_FM']    = 6e-6/u.K    #Ibarra, PRB, 50(6), 4196-4199 (1994)
        self.prop['lin_therm_exp']    = 20e-6/u.K   #Calculated from elastic constants and expansion
        self.prop['lin_therm_exp_FM'] = 12.4e-6/u.K #Calculated from elastic constants and expansion
        self.prop['Grun_c_axis']      = 0 
        self.prop['Grun_a_axis']      = 0
        self.prop['Grun_b_axis']      = 0
        self.prop['Grun_FM_c_axis']   = 0
        self.prop['Grun_FM_a_axis']   = 0
        self.prop['Grun_FM_b_axis']   = 0
        self.prop['heat_capacity']    = 335*u.J/(u.kg *u.K) #Richardson, Physics Letters A, 46(2), 153-154 (1973)
        self.prop['therm_cond']       = 50*u.W/(u.m *u.K)   #Bergman, PRB, 73(6), 1-4 (2006)
        self.prop['opt_pen_depth']    = 15*u.nm #Roughly estimated from elispsometrie
        self.prop['opt_ref_index']    = 2.24+3.88j #(800nm) Chen, PRB, 37(18), 10503-10509 (1988)
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self,name,caxis,prop):
        IronRhodium = ud.UnitCell(name, 'FeRh', caxis,**prop)
        IronRhodium.add_atom(self.Fe,0); IronRhodium.add_atom(self.Rh,0.5)
        return IronRhodium