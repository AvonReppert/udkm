# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class Pt_111_1TM:
    def __init__(self):

        self.Pt = ud.Atom('Pt')

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 2.2656*u.angstrom  # periodictable.com including fcc geometry
        self.prop['a_axis'] = 2.5818*u.angstrom  # periodictable.com including fcc geometry adjust density
        self.prop['b_axis'] = 2.5818*u.angstrom  # periodictable.com including fcc geometry adjust density
        self.prop['density'] = 21450*u.kg/(u.m**3)  # periodictable.com
        self.prop['deb_wal_fac'] = 0*u.m**2
        # (for 011) Collard, Acta metall. mater. 40, 4, 699-702 (1992)
        self.prop['elastic_c11'] = 386e9*u.kg/(u.m*u.s**2)
        # (for 011) Collard, Acta metall. mater. 40, 4, 699-702 (1992)
        self.prop['elastic_c12'] = 232e9*u.kg/(u.m*u.s**2)
        # (for 011) Collard, Acta metall. mater. 40, 4, 699-702 (1992)
        self.prop['elastic_c13'] = 232e9*u.kg/(u.m*u.s**2)
        # (for 011) Collard, Acta metall. mater. 40, 4, 699-702 (1992)
        self.prop['elastic_c33'] = 386e9*u.kg/(u.m*u.s**2)
        self.prop['sound_vel'] = 4.242*u.nm/u.ps  # "Elastic waves in solids" Royer/Dieulesaint
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = 8.9e-6/u.K  # periodictable.com
        self.prop['exp_a_axis'] = 8.9e-6/u.K  # periodictable.com
        self.prop['exp_b_axis'] = 8.9e-6/u.K  # periodictable.com
        self.prop['lin_therm_exp'] = 19.58e-6  # Calculated from elastic constants and expansion
        self.prop['Grun_c_axis'] = 0
        self.prop['Grun_a_axis'] = 0
        self.prop['Grun_b_axis'] = 0
        self.prop['heat_capacity'] = 133*u.J/(u.kg * u.K)  # periodictable.com
        self.prop['therm_cond'] = 71*u.W/(u.m * u.K)  # Duggin,Journal of Physics D: Applied Physics 3.5 (1970)
        self.prop['opt_pen_depth'] = 8*u.nm  # (800nm) below band gab
        self.prop['opt_ref_index'] = 0.5762 + 8.0776j  # (800nm) Werner, J. Phys Chem Ref. Data 38, 1013-1092 (2009)
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Pt = ud.UnitCell(name, 'Pt', caxis, **prop)
        Pt.add_atom(self.Pt, 0)
        return Pt
