# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:20:02 2024

@author: aleks
"""

import numpy as np

# Constants
h = 4.135667696e-15  # Planck's constant in eV·s
c = 3e8  # Speed of light in m/s
a = 4.078  # Lattice parameter of Au in Å
energies_keV = np.array([3, 4, 6, 8, 9, 10, 12])  # Energies in keV
miller_indices = [111, 222, 333, 444]  # Reflection orders

# Convert energies to wavelengths
energies_eV = energies_keV * 1e3  # Convert keV to eV
wavelengths = h * c / energies_eV * 1e10  # Convert wavelength to Å

# Calculate interplanar spacings


def calculate_d(hkl, lattice_param):
    h, k, l = int(hkl[0]), int(hkl[1]), int(hkl[2])
    return lattice_param / np.sqrt(h**2 + k**2 + l**2)


# Calculate diffraction angles
angles = {}
for hkl in miller_indices:
    hkl_str = str(hkl)
    d_hkl = calculate_d(hkl_str, a)
    theta_hkl = []
    for lam in wavelengths:
        sin_theta = lam / (2 * d_hkl)
        if sin_theta <= 1:  # Ensure valid angles
            theta_hkl.append(np.degrees(np.arcsin(sin_theta)))
        else:
            theta_hkl.append(None)  # No valid reflection
    angles[hkl] = theta_hkl

# Print results
print("Diffraction Angles (theta) for Au Reflections:")
print("Energies (keV):", energies_keV)
for hkl, theta_vals in angles.items():
    print(f"{hkl}: {['{:.2f}'.format(th) if th else 'None' for th in theta_vals]}")
