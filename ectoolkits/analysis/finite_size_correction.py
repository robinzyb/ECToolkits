import numpy as np
from scipy.special import erf
from scipy.linalg import toeplitz
from cp2kdata.units import au2A, au2eV
from typing import List


# these codes are adapted from the original codes of komsa
# https://cc.oulu.fi/~hkomsa19/Software.html
# reference:
# Komsa, H.-P. & Pasquarello, A. Finite-Size Supercell Correction for Charged Defects at Surfaces and Interfaces. Phys. Rev. Lett. 110, 095505 (2013).

def integer3D(f, paramcell):
    dv = np.prod(paramcell.h, dtype=float)
    x = np.sum(f)*dv
    return x

# write cell


class Paramcell:
    def __init__(self, length, divi, h):
        """
        The documentation of Paramcell class
        Parameters
        ----------
        length: list or array
            The length of the cell  in bohr
        divi: list or array
            The number of grid points in each direction
        h: float
            The grid spacing in bohr
        """
        if len(length) == 1 and isinstance(length, float):
            self.length = np.array([length, length, length])
        elif len(length) == 3:
            self.length = np.array(length)

        if divi == 0:
            self.h = h
            self.divi = np.round(self.length/self.h)
        else:
            self.divi = np.array(divi)
            self.h = self.length/self.divi
        self.divi = self.divi.astype(int)

        self.volume = np.prod(self.length)

# write charge


class GaussCharge:
    def __init__(self, Q, pos, width, paramcell, recip=False, mode='r'):
        """
        The documentation of GaussCharge class
        Parameters
        ----------
        Q: float
            The charge in e
        pos: list or array
            The position of the charge in bohr
        width: float
            The width of the Gaussian in bohr
        paramcell: Paramcell
            The cell information
        recip: bool
            If True, the reciprocal space charge density will be generated
        """

        self.Q = Q
        self.pos = pos
        self.width = width
        self.paramcell = paramcell
        self.rhocc = np.zeros(paramcell.divi)
        self.mode = 'r'

        cdisp = np.round(self.pos/paramcell.h - paramcell.divi/2)
        cdisp = cdisp.astype(int)
        drem = self.pos/paramcell.h - cdisp
        x0 = np.arange(0, paramcell.divi[0])-drem[0]
        y0 = np.arange(0, paramcell.divi[1])-drem[1]
        z0 = np.arange(0, paramcell.divi[2])-drem[2]

        x1 = np.roll(x0, cdisp[0])*paramcell.h[0]
        y1 = np.roll(y0, cdisp[1])*paramcell.h[1]
        z1 = np.roll(z0, cdisp[2])*paramcell.h[2]

        [self.X, self.Y, self.Z] = np.meshgrid(x1, y1, z1, indexing='ij')

        if not recip:
            r = np.sqrt(np.power(self.X, 2) +
                        np.power(self.Y, 2)+np.power(self.Z, 2))
            sigma = self.width
            if self.mode == 'p':
                self.rhocc = self.rhocc + \
                    np.divide(Q, r) * erf(np.divide(r, np.sqrt(2)*sigma))
            elif self.mode == 'r':
                self.rhocc = self.rhocc + np.divide(Q, np.power((sigma*np.sqrt(2*np.pi)), 3)) * np.exp(
                    np.divide(-np.power(r, 2), 2*np.power(sigma, 2)))
        else:
            print('Generate reciprocal space charge density')

            gs = 2*np.pi/paramcell.length
            gx0 = np.ceil(
                np.arange(-paramcell.divi[0]/2, paramcell.divi[0]/2)) * gs[0]
            gy0 = np.ceil(
                np.arange(-paramcell.divi[1]/2, paramcell.divi[1]/2)) * gs[1]
            gz0 = np.ceil(
                np.arange(-paramcell.divi[2]/2, paramcell.divi[2]/2)) * gs[2]
            [Gx, Gy, Gz] = np.meshgrid(gx0, gy0, gz0, indexing='ij')

            Gr = np.power(Gx, 2)+np.power(Gy, 2)+np.power(Gz, 2)

            dv = np.prod(paramcell.length, dtype=float) / \
                np.prod(paramcell.divi, dtype=float)

            rhok = Q*np.exp(-0.25*(2*np.power(self.width, 2))*Gr)
            apos = self.pos

            rhok = rhok * np.exp(-1j*(Gx*apos[0]+Gy*apos[1]+Gz*apos[2]))

            self.rhok = rhok
            self.rhocc = np.fft.ifftn(np.fft.ifftshift(rhok))/dv

            self.rhocc = self.rhocc.real
            # print(rhocc.max())


class DielProfile:
    def __init__(self,
                 z_interface_list,
                 diel_list: List[float],
                 beta_list,
                 paramcell,
                 ):
        """
        The documentation of DielProfile class
        Parameters
        ----------
        z_interface_list: list or array
            The position of the interface in bohr
        diel_list: list or array
            The dielectric constant of each layer
        beta_list: list or array
            The decay length of each layer
        paramcell: Paramcell
            The cell information
        """
        self.z_interface_list = z_interface_list

        if isinstance(diel_list, np.ndarray):
            self.diel_list_perp = diel_list
            self.diel_list_para = diel_list
            print("The gievn dielectric constant is isotropic")
            print("The dielectric constant is {}".format(self.diel_list_perp))
        elif isinstance(diel_list, dict):
            self.diel_list_perp = diel_list['perp']
            self.diel_list_para = diel_list['para']
            print("The gievn dielectric constant is anisotropic")
            print("The dielectric constant in the direction perpendicular to the interface is {}".format(
                self.diel_list_perp))
            print("The dielectric constant in the direction parallel to the interface is {}".format(
                self.diel_list_para))
        else:
            raise ValueError(
                "The type of dielectric constant is not supported, please give a list or dict")

        self.beta_list = beta_list
        self.paramcell = paramcell
        self.dielz_perp = self.gen_diel_profile(self.diel_list_perp)
        self.dielz_para = self.gen_diel_profile(self.diel_list_para)

        print("Generate dielectric profile finished")

    def gen_diel_profile(self, diel_list):
        len_z = self.paramcell.length[2]
        h = self.paramcell.h[2]
        z_gridpoints = self.paramcell.divi[2]
        z0 = np.arange(0, z_gridpoints)*h

        dielz = np.zeros(z_gridpoints)

        for k in range(z_gridpoints):
            zif = 1e6
            mif = -1
            for m in range(len(self.z_interface_list)):
                cdis = self.perdz(z0[k], self.z_interface_list[m], len_z)
                if np.abs(cdis) < np.abs(zif):
                    zif = cdis
                    mif = m

            if zif > 0:
                dielz[k] = self.ifmodel(
                    zif, diel_list[mif], diel_list[mif+1], self.beta_list[mif])
            else:
                dielz[k] = self.ifmodel(
                    zif, diel_list[mif], diel_list[mif+1], self.beta_list[mif])

        return dielz

    @staticmethod
    def ifmodel(z, diel1, diel2, beta):
        a = 0.5 * (diel2 - diel1)
        b = 0.5 * (diel2 + diel1)
        diel = a*erf(z/beta) + b
        return diel

    @staticmethod
    def perdz(z1, z2, len_z):
        dz = z1-z2
        if dz > len_z/2:
            dz = dz - len_z
        elif dz < -len_z/2:
            dz = dz + len_z
        return dz


class PBCPoissonSolver:
    def __init__(self, gauss_charge: GaussCharge, diel_profile: DielProfile, paramcell: Paramcell):
        self.gauss_charge = gauss_charge
        self.diel_profile = diel_profile
        self.paramcell = paramcell

        # neutralize the cell
        self.rho = self.gauss_charge.rhocc - self.gauss_charge.Q/self.paramcell.volume

        length = paramcell.length
        gridsize = paramcell.divi

        Gs = 2*np.pi/length
        Gx0 = np.ceil(np.arange(-gridsize[0]/2, gridsize[0]/2)) * Gs[0]
        Gy0 = np.ceil(np.arange(-gridsize[1]/2, gridsize[1]/2)) * Gs[1]
        Gz0 = np.ceil(np.arange(-gridsize[2]/2, gridsize[2]/2)) * Gs[2]
        Gx0 = np.fft.ifftshift(Gx0)
        Gy0 = np.fft.ifftshift(Gy0)
        Gz0 = np.fft.ifftshift(Gz0)

        # TODO: temporal solution, please finish this step in GaussCharge Class
        rhok = np.fft.fftn(4*np.pi*self.rho)
        self.rhok = rhok
        # rhok_tmp = self.gauss_charge.rhok
        # print((rhok_tmp-rhok).max())

        # both diel_profile.dielz_perp and diel_profile.dielz_para should be generated in DielProfile class
        dielGz_perp = np.fft.fft(self.diel_profile.dielz_perp)
        dielGz_para = np.fft.fft(self.diel_profile.dielz_para)

        LGz = len(Gz0)

        # Circular convolution matrix
        # TODO: need to be enhanced to anisotropic case
        first_row = np.concatenate(([dielGz_perp[0]], dielGz_perp[:0:-1]))
        Ag1_perp = toeplitz(dielGz_perp, first_row)/LGz

        first_row = np.concatenate(([dielGz_para[0]], dielGz_para[:0:-1]))
        Ag1_para = toeplitz(dielGz_para, first_row)/LGz

        Ag2 = np.outer(Gz0, Gz0)
        # VGz = np.zeros(LGz)
        Vk = np.zeros(rhok.shape, dtype=complex)
        # perp
        for k in range(len(Gx0)):
            for m in range(len(Gy0)):

                G_para = np.power(Gx0[k], 2) + np.power(Gy0[m], 2)

                Ag = np.multiply(Ag1_perp, Ag2) + np.multiply(Ag1_para, G_para)

                # G=0 hack!
                if (k == 0) and (m == 0):
                    Ag[0, 0] = 1

                Vk[k, m, :] = np.linalg.solve(Ag, rhok[k, m, :])
        Vk[0, 0, 0] = 0
        self.V = np.fft.ifftn(Vk).real


class UniformCharge:
    # adapted from gono's code
    def __init__(self, Q, interface_position, beta, paramcell):
        """
        The documentation of GaussCharge class
        Parameters
        ----------
        Q: float
            The charge in e
        thickness: list or array
            The position of the charge in bohr
        beta: float
            The width of the Gaussian in bohr
        paramcell: Paramcell
            The cell information
        recip: bool
            If True, the reciprocal space charge density will be generated
        """

        self.Q = Q
        self.interface_position = interface_position
        self.beta = beta
        self.paramcell = paramcell
        self.rhocc = np.zeros(paramcell.divi)

        self.rhocc = np.zeros(self.paramcell.divi)
        cell_height = self.paramcell.length[2]
        dz = self.paramcell.h[2]
        n_grid_z = self.paramcell.divi[2]
        grid_z = np.linspace(0, cell_height - dz, n_grid_z)

        charge_volume = (
            self.interface_position[1] - self.interface_position[0])
        charge_volume *= self.paramcell.length[0] * self.paramcell.length[1]

        charge_densities = [0, self.Q / charge_volume, 0]
        charge_profile = np.zeros(n_grid_z)

        for i, z in enumerate(grid_z):
            distance_closest_interface = 1.0e6
            index_closest_interface = 0

            for index, z_interface in enumerate(self.interface_position):
                distance = self.delta_z(z, z_interface, cell_height)

                if abs(distance) < abs(distance_closest_interface):
                    distance_closest_interface = distance
                    index_closest_interface = index

            charge_profile[i] = self.counter_charge_model(
                distance_closest_interface,
                charge_densities[index_closest_interface],
                charge_densities[index_closest_interface + 1],
                self.beta[index_closest_interface])

        for z_index, charge_value in enumerate(charge_profile):
            for x_index in range(paramcell.divi[0]):
                for y_index in range(paramcell.divi[1]):
                    self.rhocc[x_index, y_index, z_index] = charge_value

        # self.rhocc = self.rhocc.astype(complex)

    @staticmethod
    def counter_charge_model(z, charge_1, charge_2, width):
        a = (charge_2 - charge_1) / 2.0
        b = (charge_2 + charge_1) / 2.0
        return a * erf(z / width) + b

    @staticmethod
    def delta_z(z1, z2, periodic_cell_height):
        dz = z1 - z2
        if dz > periodic_cell_height / 2.0:
            dz -= periodic_cell_height
        if dz < -periodic_cell_height / 2.0:
            dz += periodic_cell_height
        return dz
