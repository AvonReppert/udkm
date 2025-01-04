import numpy as np
import xrayutilities as xu


def calculate_structure_factors(crystal_cif, energies, reflections):
    """
    Calculates and prints the structure factors for a given crystal and X-ray energies.

    Parameters:
        crystal_cif (str): Path to the CIF file for the material.
        energies (list): List of X-ray energies in eV.
        reflections (list): List of Miller indices for which to calculate the structure factors.
    """
    # Load the crystal structure from the CIF file
    crystal = xu.materials.material.Crystal.fromCIF(crystal_cif)

    # Header for the output
    header = (
        '--------------------------------------------------------------------------------------\n'
        ' material    |    peak    |    |F|   |   dHKL    |  theta  |   |F|^2  \n'
        '--------------------------------------------------------------------------------------'
    )
    print(header)

    for energy in energies:
        # Convert energy (eV) to wavelength (Å)
        wavelength = xu.lam2en(energy)
        print(f'         {energy} eV = {wavelength:.4f} Å\n')

        for hkl in reflections:
            # Calculate properties for the given reflection
            qvec = crystal.Q(hkl)  # Scattering vector
            F = crystal.StructureFactor(qvec, energy)  # Structure factor
            d_hkl = crystal.planeDistance(hkl)  # Interplanar spacing
            theta = np.degrees(np.arcsin(wavelength / (2 * d_hkl)))  # Bragg angle

            # Prepare the output string
            result = (
                f'{crystal.name:12} | {str(hkl):10} | {np.abs(F):8.2f} | {d_hkl:8.3f} | '
                f'{theta:8.3f} | {np.abs(F)**2:8.2f}'
            )
            print(result)
        print('-' * 86)


# Parameters
crystal_file = "Au2.cif"  # Path to the CIF file for gold
energies_eV = [8048]  # X-ray energies in eV
miller_indices = [
    [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]
]

# Call the function
calculate_structure_factors(crystal_file, energies_eV, miller_indices)
