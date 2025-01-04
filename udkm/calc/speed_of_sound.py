# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 09:02:16 2024

@author: aleks
"""

import numpy as np

# Literaturwerte aus Royer Dieulesaint p. 148

# Gegebene Werte für Aluminium
c11 = 10.73e10  # in N/m^2
c12 = 6.08e10   # in N/m^2
c44 = 2.83e10   # in N/m^2
density = 2702  # in kg/m^3


# Gegebene Werte für Platin
c11 = 34.7e10  # in N/m^2
c12 = 25.1e10   # in N/m^2
c44 = 7.65e10   # in N/m^2
density = 21400  # in kg/m^3

# Gegebene Werte für Wolfram
c11 = 52.24e10  # in N/m^2
c12 = 20.44e10   # in N/m^2
c44 = 16.06e10   # in N/m^2
density = 19260  # in kg/m^3

# Gegebene Werte für Gold
c11 = 19.25e10  # in N/m^2
c12 = 16.3e10   # in N/m^2
c44 = 4.24e10   # in N/m^2
density = 19300  # in kg/m^3


# Voigt-Mittelwerte berechnen
KV = (c11 + 2 * c12) / 3
GV = (c11 - c12 + 3 * c44) / 5

# Longitudinale und transversale Schallgeschwindigkeit
vL_V = np.sqrt((KV + (4/3) * GV) / density)
vT_V = np.sqrt(GV / density)

# Debye-Schallgeschwindigkeit berechnen
vD = (1/3 * (2/vT_V**3 + 1/vL_V**3))**(-1/3)

print(f"Voigt-Bulkmodul (KV): {KV:.2e} N/m^2")
print(f"Voigt-Schermodul (GV): {GV:.2e} N/m^2")
print(f"Longitudinale Schallgeschwindigkeit (Voigt): {vL_V:.2f} m/s")
print(f"Transversale Schallgeschwindigkeit (Voigt): {vT_V:.2f} m/s")
print(f"Debye-Schallgeschwindigkeit: {vD:.2f} m/s")
