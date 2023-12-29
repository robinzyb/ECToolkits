import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
from cp2kdata import Cp2kCube


def get_induced_charge(rho_cube_1: Cp2kCube,
                       rho_cube_2: Cp2kCube,
                       axis: str = 'z'
                       ):
    """
    Compute the induced charge density between two systems.
    Induced charge density is defined as rho_cube_2 - rho_cube_1.

    Parameters
    ----------
    rho_cube_1 : Cp2kCube
        The first charge density cube.
    rho_cube_2 : Cp2kCube
        The second charge density cube.
    axis : str, optional
        The axis along which to compute the average charge density, by default 'z'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The coordinates and the induced charge density array.

    """
    z_1, rho_1 = rho_cube_1.get_pav(axis=axis)
    z_2, rho_2 = rho_cube_2.get_pav(axis=axis)

    rho_induced = rho_2 - rho_1

    return z_1, rho_induced


def get_integrated_array(x, y):
    size = y.shape[0]
    int_array = np.zeros(size)
    for i in range(1, size+1):
        int_array[i-1] = simpson(y[:i], x[:i])
    return int_array


# get electric field
def get_micro_electric_field(x: npt.NDArray[np.float64],
                             rho: npt.NDArray[np.float64],
                             Delta_macro_Efield: float
                             ) -> npt.NDArray[np.float64]:
    """
    Calculate the micro electric field from the charge density and the macroscopic electric field difference.

    Parameters
    ----------
    x : numpy.ndarray
        The grid points where the charge density is defined.
    rho : numpy.ndarray
        The charge density in atomic units.
    Delta_macro_Efield : float
        The macroscopic electric field difference between two systems in atomic units.

    Returns
    -------
    numpy.ndarray
        The micro electric field in atomic units.

    Notes
    -----
    The micro electric field is calculated as follows:
    1. Calculate the integrand as pi * 4 * rho.
    2. Integrate the integrand over the grid points x to obtain the micro electric field.
    3. Determine a constant such that the mean of the micro electric field matches the macroscopic electric field difference.
    4. Add the constant to the micro electric field.

    The units of the input and output arrays are atomic units (au).
    """

    # atomic unit
    integrand = np.pi * 4 * rho
    micro_electric_field = get_integrated_array(x, integrand)

    # determine constant
    constant = Delta_macro_Efield - micro_electric_field.mean()

    micro_electric_field += constant
    return micro_electric_field

# get polarization


def get_micro_polarization(x: npt.NDArray[np.float64],
                           rho: npt.NDArray[np.float64],
                           Delta_macro_polarization: float
                           ) -> npt.NDArray[np.float64]:
    """
    Calculate the micro polarization from the induced charge density and the macroscopic polarization difference.

    Parameters
    ----------
    x : numpy.ndarray
        The grid points where the charge density is defined.
    rho : numpy.ndarray
        The induced charge density in atomic units.
    Delta_macro_polarization : float
        The macroscopic polarization difference between two systems in atomic units.

    Returns
    -------
    numpy.ndarray
        The micro polarization in atomic units.

    Notes
    -----
    The micro polarization is calculated as follows:
    1. Calculate the integrand as -rho.
    2. Integrate the integrand over the grid points x to obtain the micro polarization.
    3. Determine a constant such that the mean of the micro polarization matches the macroscopic polarization difference.
    4. Add the constant to the micro polarization.

    The units of the input and output arrays are atomic units (au).
    """
    # atomic unit
    # the electron charge is negative!
    integrand = -rho
    micro_polarization = get_integrated_array(x, integrand)

    # determine constant
    constant = Delta_macro_polarization - micro_polarization.mean()

    micro_polarization += constant

    return micro_polarization


def get_dielectric_susceptibility(micro_polarization, micro_electric_field):

    dielectric_susceptibility = micro_polarization / micro_electric_field
    return dielectric_susceptibility


def get_dielectric_constant(dielectric_susceptibility):
    dielectric_constant = dielectric_susceptibility*np.pi*4 + 1
    return dielectric_constant


def get_dielectric_constant_profile(rho_1, rho_2, Delta_macro_Efield, Delta_macro_polarization, axis):
    z, rho_induced = get_induced_charge(rho_1, rho_2, axis=axis)
    # electron carries negative charge
    rho_induced = -rho_induced

    micro_electric_field = get_micro_electric_field(
        z, rho_induced, Delta_macro_Efield=Delta_macro_Efield)
    micro_polarization = get_micro_polarization(
        z, rho_induced, Delta_macro_polarization=Delta_macro_polarization)
    dielectric_susceptibility = get_dielectric_susceptibility(
        micro_polarization, micro_electric_field)
    dielectric_constant = get_dielectric_constant(dielectric_susceptibility)
    return z, dielectric_constant
