# Plain implementation of equations in the paper
def get_induced_charge(rho_cube_1, rho_cube_2):
    z_1, rho_1 = rho_cube_1.get_pav()
    z_2, rho_2 = rho_cube_2.get_pav()
    #assert z_1 == z_2, "the two arrays, z_1 and z_2, must be same"
    rho_induced = rho_1 - rho_2
    # notice I multiply au2eV in cp2kdata
    rho_induced = rho_induced/au2eV
    # electron carries negative charge
    rho_induced = -rho_induced 
    # notice I multiply au2A in cp2kdata
    z_1 = z_1/au2A

    return z_1, rho_induced

def get_integrated_array(x, y):
    size = y.shape[0]
    int_array = np.zeros(size)
    for i in range(1, size+1):
        int_array[i-1] = simpson(y[:i], x[:i])
    return int_array


# get electric field
def get_micro_electric_field(x, rho, Delta_macro_Efield):
    """
    in atomic units (au)
    Delta_macro_Efield is the macroscopic electric field difference between two systems
    Delat_macro_Efield (au)
    """

    
    # atomic unit
    # the electron charge is negative!
    integrand = np.pi * 4 * rho
    micro_electric_field = get_integrated_array(x, integrand)

    # determine constant
    constant = Delta_macro_Efield - micro_electric_field.mean()
    
    micro_electric_field += constant
    return micro_electric_field
    
# get polarization

def get_micro_polarization(x, rho, Delta_macro_polarization):
    """
    rho: induced charge density in atomic units (au)
    
    Delta_macro_polarization is the macroscopic polarization difference between two systems
    
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


def get_dielectric_constant_profile(rho_1, rho_2, Delta_macro_Efield, Delta_macro_polarization):
    z, rho_induced = get_induced_charge(rho_1, rho_2)
    micro_electric_field = get_micro_electric_field(z, rho_induced, Delta_macro_Efield=Delta_macro_Efield)
    micro_polarization = get_micro_polarization(z, rho_induced, Delta_macro_polarization=Delta_macro_polarization)
    dielectric_susceptibility = get_dielectric_susceptibility(micro_polarization, micro_electric_field)
    dielectric_constant = get_dielectric_constant(dielectric_susceptibility)
    return z, dielectric_constant
