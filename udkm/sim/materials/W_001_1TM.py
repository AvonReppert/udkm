# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class W_001_1TM:
    def __init__(self):

        self.W = ud.Atom('W')

        self.prop = {}
        self.prop['crystal_struc'] = 'bcc'
        self.prop['c_axis'] = 3.1652*u.angstrom  # periodictable.com
        self.prop['a_axis'] = 3.1652*u.angstrom  # periodictable.com
        self.prop['b_axis'] = 3.1652*u.angstrom  # periodictable.com
        self.prop['density'] = 19250*u.kg/(u.m**3)  # periodictable.com
        self.prop['deb_wal_fac'] = 0*u.m**2
        # (for 001 @300K) Featherston, Physical Review 130, 4, 1324-1333 (1963)
        self.prop['elastic_c11'] = 523.3e9*u.kg/(u.m*u.s**2)
        # (for 001 @300K) Featherston, Physical Review 130, 4, 1324-1333 (1963)
        self.prop['elastic_c12'] = 204.5e9*u.kg/(u.m*u.s**2)
        # (for 001 @300K) Featherston, Physical Review 130, 4, 1324-1333 (1963)
        self.prop['elastic_c13'] = 204.5e9*u.kg/(u.m*u.s**2)
        # (for 001 @300K) Featherston, Physical Review 130, 4, 1324-1333 (1963)
        self.prop['elastic_c33'] = 523.3e9*u.kg/(u.m*u.s**2)
        self.prop['sound_vel'] = 5.212*u.nm/u.ps  # Calculated from c11 and density
        self.prop['phonon_damping'] = 0*u.kg/u.s
        self.prop['exp_c_axis'] = 4.6e-6/u.K  # Nix, Physical Review 61, 74-78 (1942)
        self.prop['exp_a_axis'] = 4.6e-6/u.K  # Nix, Physical Review 61, 74-78 (1942)
        self.prop['exp_b_axis'] = 4.6e-6/u.K  # Nix, Physical Review 61, 74-78 (1942)
        self.prop['lin_therm_exp'] = 8.19e-6  # Calculated from elastic constants and expansion
        self.prop['Grun_c_axis'] = 0
        self.prop['Grun_a_axis'] = 0
        self.prop['Grun_b_axis'] = 0
        # White, Journal of Physical and Chemical Reference Data 13, 1251 (1984)
        self.prop['heat_capacity'] = 133*u.J/(u.kg * u.K)
        # Hust, (No. PB-84-235878; NBSIR-84/3007). National Bureau of Standards, Boulder, CO (USA). Chemical Engineering Science Div. (1984)
        self.prop['therm_cond'] = 170*u.W/(u.m * u.K)
        self.prop['opt_pen_depth'] = 8*u.nm  # (800nm)
        # (800nm) Adachi, The Handbook on optical constants of metals (2012) p.277
        self.prop['opt_ref_index'] = 3.56 + 2.73j
        self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        W = ud.UnitCell(name, 'W', caxis, **prop)
        W.add_atom(self.W, 0)
        W.add_atom(self.W, 0.5)
        return W
