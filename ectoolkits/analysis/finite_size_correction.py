import numpy as np
from scipy.special import erf
from scipy.linalg import toeplitz
from cp2kdata.units import au2A, au2eV




# these codes are adapted from the original codes of komsa
# https://cc.oulu.fi/~hkomsa19/Software.html
# reference: 
# Komsa, H.-P. & Pasquarello, A. Finite-Size Supercell Correction for Charged Defects at Surfaces and Interfaces. Phys. Rev. Lett. 110, 095505 (2013).
  

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
            r = np.sqrt(np.power(self.X, 2)+np.power(self.Y, 2)+np.power(self.Z, 2))
            sigma = self.width
            if self.mode == 'p':
                self.rhocc = self.rhocc + np.divide(Q, r) * erf(np.divide(r, np.sqrt(2)*sigma))
            elif self.mode == 'r':
                self.rhocc = self.rhocc + np.divide(Q,np.power((sigma*np.sqrt(2*np.pi)), 3)) * np.exp(np.divide(-np.power(r, 2), 2*np.power(sigma, 2)))
        else:
            print('Generate reciprocal space charge density')

            gs = 2*np.pi/paramcell.length
            gx0 = np.ceil(np.arange(-paramcell.divi[0]/2, paramcell.divi[0]/2)) * gs[0]
            gy0 = np.ceil(np.arange(-paramcell.divi[1]/2, paramcell.divi[1]/2)) * gs[1]
            gz0 = np.ceil(np.arange(-paramcell.divi[2]/2, paramcell.divi[2]/2)) * gs[2]
            [Gx, Gy, Gz] = np.meshgrid(gx0, gy0, gz0, indexing='ij')

            Gr = np.power(Gx, 2)+np.power(Gy, 2)+np.power(Gz, 2)

            dv = np.prod(paramcell.length, dtype=float)/np.prod(paramcell.divi, dtype=float)
    
            rhok = Q*np.exp(-0.25*(2*np.power(self.width, 2))*Gr)
            apos = self.pos

            rhok = rhok * np.exp(-1j*(Gx*apos[0]+Gy*apos[1]+Gz*apos[2]))

            self.rhok = rhok
            self.rhocc = np.fft.ifftn(np.fft.ifftshift(rhok))/dv
            #print(rhocc.max())
 

class DielProfile:
    def __init__(self, z_interface_list, diel_list, beta_list, paramcell):
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
        self.diel_list = diel_list
        self.beta_list = beta_list
        self.paramcell = paramcell
        self.dielz = self.gen_diel_profile()

    def gen_diel_profile(self):
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
                dielz[k] = self.ifmodel(zif, self.diel_list[mif], self.diel_list[mif+1], self.beta_list[mif])
            else:
                dielz[k] = self.ifmodel(zif, self.diel_list[mif], self.diel_list[mif+1], self.beta_list[mif])

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
        self.rho = self.gauss_charge.rhocc.real - self.gauss_charge.Q/self.paramcell.volume


        length = paramcell.length
        gridsize = paramcell.divi

        Gs = 2*np.pi/length
        Gx0 = np.ceil(np.arange(-gridsize[0]/2, gridsize[0]/2)) * Gs[0]
        Gy0 = np.ceil(np.arange(-gridsize[1]/2, gridsize[1]/2)) * Gs[1]
        Gz0 = np.ceil(np.arange(-gridsize[2]/2, gridsize[2]/2)) * Gs[2]
        Gx0 = np.fft.ifftshift(Gx0)
        Gy0 = np.fft.ifftshift(Gy0)
        Gz0 = np.fft.ifftshift(Gz0)

       
        #TODO: temporal solution, please finish this step in GaussCharge Class
        rhok = np.fft.fftn(4*np.pi*self.rho)
        self.rhok = rhok
        #rhok_tmp = self.gauss_charge.rhok
        #print((rhok_tmp-rhok).max())
        
        dielGz = np.fft.fft(self.diel_profile.dielz)


        LGz = len(Gz0)

        # Circular convolution matrix
        #TODO: need to be enhanced to anisotropic case 
        first_row = np.concatenate(([dielGz[0]], dielGz[:0:-1]))
        Ag1 = toeplitz(dielGz, first_row)/LGz
        #self.Ag1 = Ag1
        Ag2 = np.outer(Gz0, Gz0)
        #VGz = np.zeros(LGz)
        Vk = np.zeros(rhok.shape, dtype=complex)

        for k in range(len(Gx0)):
            for m in range(len(Gy0)):

                Gpars = np.power(Gx0[k], 2) + np.power(Gy0[m], 2)
                
                Ag = np.multiply(Ag1, (Ag2 + Gpars))
                
                # G=0 hack!
                if (k == 0) and (m == 0):
                    Ag[0, 0] = 1

                Vk[k, m, :] = np.linalg.solve(Ag, rhok[k, m, :])
        Vk[0, 0, 0] = 0
        self.V = np.fft.ifftn(Vk).real

        


