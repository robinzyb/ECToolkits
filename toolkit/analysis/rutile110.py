import os
import numpy as np

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import (distance_array,
                                      apply_PBC,
                                      minimize_vectors)
from ..utils.rutile110 import (count_cn, 
                               cellpar2volume)
from ..utils.utils import create_path

import numpy as np
import matplotlib.pyplot as plt


class WatDensity(AnalysisBase):
    """MDAnalysis class calculating z-axis averaged water densities 
       for slab-water interface model. Outputs number densities for both
       hydrogens and water.

    Args:
        AnalysisBase (object): MDAnalysis Analysis class base
    """
    
    def __init__(self, atomgroup, M='Ti', rotM=None, dz_bin=0.1, cutoff=2.8):
        """Init water density analysis

        Args:
            atomgroup (MDAnalysis.atomgroup): 
                obtained from MDAnalysis.Universe.atoms
            M (str, optional): 
                Metal element in rutile strucure. Defaults to 'Ti'.
            rotM (numpy.ndarray, optional): 
                3*3 2d rotation matrix. Provide it if Z-axis of the simulation 
                cell isn't parallel to \hkl(110). Defaults to None.
            dz_bin (float, optional): 
                step size for binning the Z-axis. Defaults to 0.1 angstrom.
            cutoff (float, optional): 
                cutoff distance for calculating coordination number of M-O. Make 
                sure you could use this value sucessfully select all the water oxygen
                index, that is exactly m oxygen. (TiO2)_n (H2O)_m. For TiO2
                interface, 2.8 angstrom works well.
                Defaults to 2.8 angstrom.
        
        Usage example:
            Please wait...
        """
        
        self._ag    = atomgroup
        self.rotM   = rotM
        self.M      = M
        self.dz_bin = dz_bin
        self.cutoff = cutoff
        
        trajectory  = atomgroup.universe.trajectory
        super(WatDensity, self).__init__(trajectory)
        
        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")
        
        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)
        
    def _prepare(self):
        #------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.volume = cellpar2volume(self.cellpar)
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements==self.M)[0]
        self.idx_O = np.where(self._ag.elements=='O')[0]
        self.idx_H = np.where(self._ag.elements=='H')[0]
        
        self.watOidx, self.watHidx = self.get_watOidx()
        # Note: there is a trick here
        # All ase generated edge 1-11 slab that 4-coordinated Ti of lower surface edge
        # has index 1.
        # Translate this titanium ion to self._transtarget will yeild slab looking like
        """
        ooooooooooooooooooooooooooooo
        ooooooooooooooooooooooooooooo  [110]
        ooooooooooooooooooooooooooooo  ^ 
        **************************000  |             o: TiO2 slab
        ***************************00  |             0: <1-11>-edge
        *****************************  -----> [001]  *: water
        *****************************
        *****************************
        *****************************
        *****************************
        *****************************
        *****************************
        *****************************
        *****************************
        00***************************
        000**************************
        ooooooooooooooooooooooooooooo
        ooooooooooooooooooooooooooooo
        ooooooooooooooooooooooooooooo
        """
        # such that the water are centered in the box
        self.d_halfslab = (self.xyz[:, -1][self.idx_M].max() - self.xyz[:, -1][self.idx_M].min())/2
        self._idxanchor = 1
        self._transtarget = np.array([0.3, 0.3, -self.d_halfslab])
        
        #---------------------------- prepare results array ----------------------------
        # guess the water film thickness
        self.film_thickness = self.get_wat_thickness()
        self.z_bins = np.arange(0, self.film_thickness + self.dz_bin, self.dz_bin)
        self.nbins = self.z_bins.shape[0] - 1
        
        self.hist_oxygen   = np.empty((self.n_frames, self.nbins), dtype=np.float32)
        self.hist_hydrogen = np.empty((self.n_frames, self.nbins), dtype=np.float32)
              
    def _single_frame(self):
        # The calculation
        xyz = self.update_pos(self._ag)
        hist_o, hist_h = self.get_z_density(self._ag)
        
        # save results
        self.hist_oxygen[self._frame_index]   = hist_o
        self.hist_hydrogen[self._frame_index] = hist_h

    def _conclude(self):
        #--------------------- save the original data (frame-wise) ---------------------
        np.save(os.path.join(self.datdir, "hist_oxygen.npy"),   self.hist_oxygen)
        np.save(os.path.join(self.datdir, "hist_hydrogen.npy"), self.hist_hydrogen)
        
        #----------------------- calculate time-averaged density -----------------------
        dat_z = self.z_bins[:-1] + self.dz_bin/2
        dV = self.volume*(self.film_thickness/self.cellpar[2]) / self.nbins
        const = self.number_density2watdensity(dV)
        self.hist_oxygen *= const
        self.hist_hydrogen *= const
        
        dat_oxygen =   np.array([dat_z, self.hist_oxygen.mean(axis=0)]).T
        dat_hydrogen = np.array([dat_z, self.hist_hydrogen.mean(axis=0)]).T
        np.savetxt(os.path.join(self.datdir, "oxygen.dat"), dat_oxygen, header="z\tdensity", fmt="%2.6f")
        np.savetxt(os.path.join(self.datdir, "hydrogen.dat"), dat_hydrogen, header="z\tdensity", fmt="%2.6f")
        np.savetxt(os.path.join(self.datdir, "hist2rho_scale.dat"), np.array([const]))
        
        #-------------------------------- plot density ---------------------------------
        self.plot_density_z(dat_oxygen, dat_hydrogen)
        
    def get_watOidx(self):
        """This methods would select all water oxygen index

        Returns:
            tuple: (<water oxygen index>, <all Hydrogen index>). Note, for rutile-\hkl(110)
            water interface, all hydrogen comes from water.
        """
        xyz = self.xyz
        cn_H = count_cn(xyz[self.idx_O, :], xyz[self.idx_H, :], 1.2, None, self.cellpar)
        cn_M = count_cn(xyz[self.idx_O, :], xyz[self.idx_M, :], self.cutoff, None, self.cellpar)
        watOidx = self.idx_O[(cn_H >= 0) * (cn_M <= 1)]
        watHidx = self.idx_H
        return watOidx, watHidx
        
    def get_wat_thickness(self):
        trans = self._transtarget - self.xyz[self._idxanchor]
        xyz = apply_PBC(self.xyz+trans, self.cellpar)
        if self.rotM is not None:
            xyz = np.matmul(xyz, self.rotM)
        thickness = xyz[:, -1][self.watOidx].max() - xyz[:, -1][self.watOidx].min()
        thickness = np.round(thickness)+2
        return thickness
    
    def update_pos(self, ag):
        """To avoding collective slab drifting during the simulation. Move atom with index 'self._idxanchor'
        to the same position each frame. Defaults to 1 (One of the slab atom if you are using ase for modle
        generation). 
        """
        trans = self._transtarget - ag.positions[self._idxanchor]
        ag.positions += trans
        return ag.positions
    
    def get_z_density(self, ag):
        xyz = apply_PBC(ag.positions, box=self.cellpar)
        if self.rotM is not None:
            z = np.matmul(xyz, self.rotM)[:, -1] 
        else:
            z = xyz[:, -1]
            
        z_watO = z[self.watOidx] - z[self.watOidx].mean()
        z_watH = z[self.watHidx] - z[self.watHidx].mean()

        zbins = self.z_bins - self.z_bins.mean()
        hist_o, _ = np.histogram(z_watO, bins=zbins)
        hist_h, _ = np.histogram(z_watH, bins=zbins)
        return hist_o, hist_h
    
    def plot_density_z(self, oxygen, hydrogen):
        fig = plt.figure(figsize=(8, 5), dpi=200)
        ax = plt.gca()
        ax.plot(oxygen[:, 0],   oxygen[:, 1], color="r", label="O")
        ax.plot(hydrogen[:, 0], hydrogen[:, 1], color="b", label="H")
        ax.axhline(y=1, color='r', ls='--')
        ax.axhline(y=2, color='b', ls='--')
        ax.set_xlabel(r"Relative z [$\mathsf{\AA}$]")
        ax.set_ylabel(r"Number density [$\mathsf{g\,cm^{-3}}$]")
        ax.legend(loc="upper center")
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.figdir, "density.pdf"), format="PDF")
        
    @staticmethod
    def number_density2watdensity(dV):
        """dV is the bin volume. Will return a constant converting number density to water mass density.
        The unit is g cm^{-3}
        """
        ag2cm = 1E-8
        NA = 6.02214076*1E23
        M_wat = 18.02/NA
        dV = dV*ag2cm**3
        return M_wat/dV


class RutileDisDeg(AnalysisBase):
    """MDAnalysis class calculating surface water dissociation degree for rutile \hkl(110)-water interface.
    Besides dissociation, this method will also output surface adsorption water oxygen index, which is useful for 
    TiO2-water interface, because adsorbed water in this system sometimes exchange with sub-interface water.  

    Args:
        AnalysisBase (object): MDAnalysis Analysis class base
    """
    
    def __init__(self, atomgroup, owidx, cn5idx, edge4idx=None, edge5idx=None, M='Ti', cutoff=2.8):
        """Initialising a dissociation degree calculating class

        Args:
            atomgroup (MDAnalysis.Atomgroup): 
                Just use all the atoms in your universe. universe.atoms
            owidx (np.ndarray): 
                1-d integer array, which contains the indicies for water oxygen in your inteface model.
                you can use medthod 'get_watOidx' to get this index.
            cn5idx (np.ndarray): 
                2-d integer array, which contains all the indicies for terrace Ti_{5c} atoms in your 
                interface model. This can be obtained using Method 'Rutile1p11Edge' or 'Rutile110'.
            edge4idx (np.ndarray, optional): 
                2-d integer array, which contains all the indicies for edge Ti_{4c} atoms in your 
                interface model. This option is specially tailored for \hkl<1-11> edge model. You don't need
                to specify this. Defaults to None.
            edge5idx (np.ndarray, optional): 
                2-d integer array, which contains all the indicies for edge Ti_{5c} atoms in your 
                interface model. This option is specially tailored for \hkl<1-11> edge model. You don't need
                to specify this. Defaults to None.
            M (str, optional): 
                Metal element in rutile strucure. Defaults to 'Ti'.
            cutoff (float, optional): 
                cutoff distance for calculating coordination number of M-O. For TiO2 \hkl(110)-water interface, 
                2.8 angstrom works well. This value is used for determining adsorbed water above terrace Ti5c 
                atoms. Distances between water oxygen and Ti5c lager than `cutoff` will consider as **not** 
                adsorbed. Defaults to 2.8 angstrom.
        
        Usage example:
            Please wait...
        """   
        # load inputs 
        self._ag = atomgroup
        self.owidx = owidx
        self.upper_cn5idx, self.lower_cn5idx = cn5idx
        self.n_cn5idx = cn5idx.shape[-1]
        if edge4idx is not None:
            self.upper_edge4idx, self.lower_edge4idx = edge4idx
            self.n_edge4idx = edge4idx.shape[-1]
        else:
            self.n_edge4idx = 0
        if edge5idx is not None:
            self.upper_edge5idx, self.lower_edge5idx = edge5idx
            self.n_edge5idx = edge5idx.shape[-1]
        else:
            self.n_edge5idx = 0
        self.M = M
        self.cutoff = cutoff
        
        # MDA analysis class routine
        trajectory = atomgroup.universe.trajectory
        super(RutileDisDeg, self).__init__(trajectory)
        
        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")
        
        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)
        
        # the name for output files
        self.fn_adind_cn5   = os.path.join(self.datdir, "ad_O_indicies.npy")
        self.fn_adind_edge5 = os.path.join(self.datdir, "ad_O_indicies-edge5.npy")
        self.fn_adind_edge4 = os.path.join(self.datdir, "ad_O_indicies-edge4.npy")
        self.fn_cn          = os.path.join(self.datdir, "SurfaceOxygenCN.npy")
        self.fn_disdeg      = os.path.join(self.datdir, "disdeg.npy")
        
    def _prepare(self):
        #------------------------ initialize usefule constants -------------------------        
        self.cellpar = self._ag.dimensions
        self.xyz     = self._ag.positions
        self.idx_M   = np.where(self._ag.elements==self.M)[0]
        self.idx_O   = np.where(self._ag.elements=='O')[0]
        self.idx_H   = np.where(self._ag.elements=='H')[0]
        
        #-------------------------- prepare temporary arraies --------------------------
        self.dm_cn5   = np.empty((self.n_cn5idx, self.owidx.shape[0]), dtype=float)
        self.dm_edge4 = np.empty((self.n_edge4idx, self.owidx.shape[0]), dtype=float)
        self.dm_edge5 = np.empty((self.n_edge5idx, self.owidx.shape[0]), dtype=float)
        
        #---------------------------- prepare results array ----------------------------
        self._n          = self.n_cn5idx + self.n_edge4idx*2 + self.n_edge5idx
        self.ad_indicies = np.empty((self.n_frames, 2, self._n), dtype=int)
        self.cn          = np.empty((self.n_frames, 2, self._n), dtype=float)
        self.disdeg      = np.empty((self.n_frames, 2, 4), dtype=float)
    
    
    def _single_frame(self):
        # get 'neighbor' (Adsorbed) oxygen indicies
        # indicies being -1 means no adsorbed oxygen atoms
        # within a cutoff sphere (self.cutoff)
        self.ad_indicies[self._frame_index, 0, :self.n_cn5idx] = \
            self.get_neighbor_oxygen(self.upper_cn5idx, self.dm_cn5, n_ow=1)
        self.ad_indicies[self._frame_index, 1, :self.n_cn5idx] = \
            self.get_neighbor_oxygen(self.lower_cn5idx, self.dm_cn5, n_ow=1)
        if self.n_edge5idx > 0:
            self.ad_indicies[self._frame_index, 0, self.n_cn5idx:(self.n_cn5idx+self.n_edge5idx)] = \
                self.get_neighbor_oxygen(self.upper_edge5idx, self.dm_edge5, n_ow=1)
            self.ad_indicies[self._frame_index, 1, self.n_cn5idx:(self.n_cn5idx+self.n_edge5idx)] = \
                self.get_neighbor_oxygen(self.lower_edge5idx, self.dm_edge5, n_ow=1)
        if self.n_edge4idx > 0:
            self.ad_indicies[self._frame_index, 0, (self.n_cn5idx+self.n_edge5idx):] = \
                self.get_neighbor_oxygen(self.upper_edge4idx, self.dm_edge4, n_ow=2)       
            self.ad_indicies[self._frame_index, 1, (self.n_cn5idx+self.n_edge5idx):] = \
                self.get_neighbor_oxygen(self.upper_edge4idx, self.dm_edge4, n_ow=2)
            
        # calculate the coordination numnber for the oxygen
        idx_o = self.ad_indicies[self._frame_index].flatten()
        mask = (idx_o==-1)   # mask using -1 indicies
        cn = count_cn(self._ag.positions[idx_o], self._ag.positions[self.idx_H], 
                      cutoff_hi=1.2, cutoff_lo=None, cell=self.cellpar).astype(float)
        cn[mask] = np.nan
        self.cn[self._frame_index] = cn.reshape(2, self._n)

    def _conclude(self):
        res_cn5_Oind    = self.ad_indicies[:, :, :self.n_cn5idx]
        np.save(self.fn_adind_cn5, res_cn5_Oind)
        if self.n_edge5idx > 0:
            res_edge5_Oind  = self.ad_indicies[:, :, self.n_cn5idx:(self.n_cn5idx+self.n_edge5idx)]
            np.save(self.fn_adind_edge5, res_edge5_Oind)
        if self.n_edge4idx > 0:
            res_edge4_Oind  = \
                self.ad_indicies[:, :, (self.n_cn5idx+self.n_edge5idx):].reshape(self.n_frames,
                                                                                 2,
                                                                                 self.n_edge4idx,
                                                                                 2)
            np.save(self.fn_adind_edge4, res_edge4_Oind)
        # save coordination numbers
        np.save(self.fn_cn, self.cn)
        # cal disdeg and save
        n_effect = np.count_nonzero(~np.isnan(self.cn), axis=-1)
        for kk in range(4):
            self.disdeg[:, :, kk] = \
                np.nansum(self.cn==kk, axis=-1)/n_effect
        np.save(self.fn_disdeg, self.disdeg)
        return None
        
    def get_neighbor_oxygen(self, idx_ti, res_dm, n_ow=1):
        """Give a group of ti atoms, find their neighboring water oxygen within cutoff rasius `self.cutoff`.
        Returns water oxygen indicies

        Args:
            idx_ti (np.ndarray): 
                1-d integer array containing indicies for Ti5c atoms.
            res_dm (np.ndarray): 
                2-d distance array containing pair-wise distance between Ti and all water oxygen atoms. 
                This array is prepared in `_prepare`.
            n_ow (int, optional): 
                number of neighboring oxygen for input group of Ti atoms. For example, Ti5c has 1 ad-water; 
                and edge Ti4c has 2 ad-water. Defaults to 1.

        Returns:
            np.ndarray: 
                1-d array containing adsorbed water oxygen indicies. If these is no water oxygen within the 
                cutoff raidius, a masking value of '-1' is provided
        """
        # group_idx 
        distance_array(self._ag.positions[idx_ti], self._ag.positions[self.owidx], result=res_dm, box=self.cellpar)
        sort_idx = np.argsort(res_dm, axis=1)
        res_idx = self.owidx[sort_idx[:, :n_ow]]
        mask = (np.take_along_axis(res_dm, sort_idx[:, :n_ow], axis=1)>self.cutoff)
        res_idx[mask] = -1
        return res_idx.reshape(-1)

class dAdBridge(AnalysisBase):
    """MDAnalysis class calculating distances between adsobed water oxygen atoms and two neighboring bridge 
    oxygen atoms. The oxygen water oxygen adsorb on terrace Ti5c atoms.
    Args:
        AnalysisBase (object): MDAnalysis Analysis class base
    """
    def __init__(self, atomgroup, idx_cn5, idx_adO, idx_obr, ref_vec=None, M='Ti'):
        """Initialize analysis method 'dAdBridge'

        Args:
            atomgroup (MDAnalysis.Atomgroup): 
                Just use all the atoms in your universe. universe.atoms
            idx_cn5 (np.ndarray): 
                2-d integer array, with shape (2, n_cn5), which contains the upper and lower indicies for 
                terrace Ti_{5c} atoms in your interface model. This can be obtained using Method 'Rutile1p11Edge' 
                or 'Rutile110'.
            idx_adO (np.ndarray):  
                3-d integer array, with shape (n_frames, 2, n_cn5), which contains the indicies of adsorbed 
                water oxygen atoms at every MD snapshot. This array is obtained using MDAnalsis analysis 
                class: RutileDisDeg.
            idx_obr (np.ndarray): 
                3-d integer array, with shape (2, n_cn5, 2), which contains the upper and lower indicies for 
                two rows of bridge oxygen indicies corresponding to `idx_cn5`. This can be obtained using Method 
                'pair_M5c_n_obr'. 
            ref_vec (np.ndarray, optional): 
                reference vector v_r. The horizontal distances between adsorption oxygen and pairing bridge oxygen
                is obtained by dot product
                                                np.dot(v_r, v(Oad-Obr)).
                If `ref_vec` is not provided, the normalized vector of averaged v(Obr1-Obr2) will be used.
                Defaults to None.
            M (str, optional): 
                Metal element in rutile strucure. Defaults to 'Ti'.. Defaults to 'Ti'.

        Examples:
            Please wait ...
        """
        # load inputs 
        self._ag = atomgroup
        self.idx_cn5 = idx_cn5
        self.idx_adO = idx_adO
        self.idx_obr = idx_obr
        self.idx_obr1 = np.array([idx_obr[0, :, 0], idx_obr[1, :, 0]])
        self.idx_obr2 = np.array([idx_obr[0, :, 1], idx_obr[1, :, 1]])
        
        if ref_vec is None:
            self.ref_vec = None
            print("'ref_vec' is not provided. Use auto generated reference vectors.")
        else:
            self.ref_vec = ref_vec    
        self.M = M
        
        # MDA analysis class routine
        trajectory = atomgroup.universe.trajectory
        super(dAdBridge, self).__init__(trajectory)
        
        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")
        
        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)
        
        # the file names for output data
        self.fn_upperdab = os.path.join(self.datdir, "upper-dab.npy")
        self.fn_lowerdab = os.path.join(self.datdir, "lower-dab.npy")
    
    def _prepare(self):
        #------------------------ initialize usefule constants -------------------------        
        self.cellpar = self._ag.dimensions
        self.xyz     = self._ag.positions
        self.idx_M   = np.where(self._ag.elements==self.M)[0]
        self.idx_O   = np.where(self._ag.elements=='O')[0]
        self.idx_H   = np.where(self._ag.elements=='H')[0]
        if self.ref_vec is None:
            ref_upper    = self.get_ref(self.idx_obr[0][:, 0], self.idx_obr[0][:, 1])
            ref_lower    = self.get_ref(self.idx_obr[1][:, 0], self.idx_obr[1][:, 1])
            self.ref_vec = (ref_upper + ref_lower)/2
        
        #---------------------------- prepare results array ----------------------------
        self.dab = np.empty((self.n_frames, 2, 2, self.idx_cn5.shape[-1]), dtype=float)
        
    def _single_frame(self):
        upper_v1, upper_v2 = self.get_dab(self.idx_adO[self._frame_index, 0, :],
                                          self.idx_obr1[0], self.idx_obr2[0])
        lower_v1, lower_v2 = self.get_dab(self.idx_adO[self._frame_index, 1, :],
                                          self.idx_obr1[1], self.idx_obr2[1])
        self.dab[self._frame_index, 0, :, ] = np.array([upper_v1, upper_v2])
        self.dab[self._frame_index, 1, :, ] = np.array([lower_v1, lower_v2])
        
    
    def _conclude(self):
        np.save(self.fn_upperdab, self.dab[:, 0, :, :])
        np.save(self.fn_lowerdab, self.dab[:, 1, :, :])

    
    def get_ref(self, idx1, idx2):
        """use minimum image vector between two rows of obr as reference vectors"""
        xyz = self._ag.positions
        v_mic = minimize_vectors(xyz[idx1] - xyz[idx2], box=self.cellpar)
        return np.mean(v_mic/np.linalg.norm(v_mic, axis=1)[:, np.newaxis], axis=0)
    
    def get_dab(self, idx_adO, idx_obr1, idx_obr2):
        """get distances between Adsorption water oxygen and Bridge oxygen, aka, dAB.

        Args:
            idx_adO (np.ndarray): 
                indicies of ad water oxygen atoms. Use -1 as masking marker, meaning didn't found Obr
                whthin cutoff. (see analysis class `RutileDisDeg`)
            idx_obr1 (np.ndarray): 
                indicies of row#1 of bridge oxygen atoms. 
            idx_obr2 (np.ndarray): 
                indicies of row#2 of bridge oxygen atoms.

        Returns:
            tuple (np.ndarray, np.ndarray):
                (<result distances: Oad-Obr#1>, <result distances: Oad-Obr#2>)
        """
        # Remember that idx_adO being -1 just means this Oad is not found 
        # within a cutoff sphere. Therefore, first prepare mask, then 
        # substituting the masked value with 'np.nan'
        mask = (idx_adO == -1)    
        xyz = self._ag.positions
        
        micv1 = minimize_vectors(xyz[idx_adO] - xyz[idx_obr1], box=self.cellpar)
        micv2 = minimize_vectors(xyz[idx_adO] - xyz[idx_obr2], box=self.cellpar)
        
        res1  = np.matmul(micv1, self.ref_vec)
        res2  = np.matmul(micv2, self.ref_vec)
        res1[mask] = np.nan
        res2[mask] = np.nan
        return res1, res2
