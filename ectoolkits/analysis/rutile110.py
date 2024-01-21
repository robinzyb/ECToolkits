import os
import numpy as np

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import (calc_bonds,
                                      calc_angles,
                                      distance_array,
                                      apply_PBC,
                                      minimize_vectors)

from ..utils.rutile110 import (count_cn,
                               cellpar2volume,
                               get_watOidx)
from ..utils.utils import create_path

import numpy as np
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # N O T I C E # # # # # # # # # # # # # # # # # #
# All of the following analysis methods are child classes for
# MDAnalysis's AnalysisBase. They all use MDAnalysis's atomgroup object as input.
# Below are some ways for you to properly prepare such atomgroup objects:
#    >>> from MDAnalysis import Universe
#    >>> from ase.io import read
# 1. from a xyz file "traj.xyz"
#    >>> atoms = read("init.cif")
#    >>> u = Universe("traj.xyz")
#    >>> u.dimensions = atoms.cell.cellpar()   # Required. Provide cell info for NVT traj.
#    >>> u.trajectory.ts.dt = 0.00005          # Optional. To suppress a warning.
#    >>> ag = u.atoms
# 2. from a pickled universe file "traj.uni"
#    >>> import pickle
#    >>> atoms = read("init.cif")
#    >>> with open("traj.uni", 'rb') as f:
#    ...     u = pickle.load(f)
#    >>> u.dimensions = atoms.cell.cellpar()   # Required. Provide cell info for NVT traj.
#    >>> u.trajectory.ts.dt = 0.00005          # Optional. To suppress a warning.
#    >>> ag = u.atoms
# 3. from a AMBER netCDF file "traj.ncdf"
#    >>> atoms = read("init.cif")
#    >>> u = Universe("traj.ncdf")
#    >>> u.add_TopologyAttr("elements", np.array(atoms.get_chemical_symbols()))
#                                              # Required. AMBER netCDF trajs don't have element info
#    >>> u.dimensions = atoms.cell.cellpar()   # Required. Provide cell info for NVT traj.
#    >>> u.trajectory.ts.dt = 0.00005          # Optional. To suppress a warning.
#    >>> ag = u.atoms
# # # # # # # # # # # # # # # # # # N O T I C E # # # # # # # # # # # # # # # # # #


class WatDensity(AnalysisBase):
    """MDAnalysis class calculating z-axis averaged water densities
       for slab-water interface model. Outputs number densities for both
       hydrogens and water.

    Args:
        AnalysisBase (object): MDAnalysis Analysis class base

    Usage example:
    (1) Rutile (110)-water interface with <1-11> edge
        ```python
        from toolkit.utils.rutile110 import get_rotM

        vecy = np.array([26.34844236,  1.8642182,  -2.0615678])
        vecz = np.array([0.49339,  -0.070221,  9.201458])
        rotM = get_rotM(vecy, vecz)

        wd = WatDensity(ag, rotM=rotM)
        wd.run()
        ```
    (2) Flat rutile
        ```python
        wd = WatDensity(ag, rotM=None)
        wd.run()
        ```
    """

    def __init__(self, atomgroup, M='Ti', rotM=None, dz_bin=0.1, cutoff=2.8):
        r"""Init water density analysis

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
        """

        self._ag = atomgroup
        self.rotM = rotM
        self.M = M
        self.dz_bin = dz_bin
        self.cutoff = cutoff

        trajectory = atomgroup.universe.trajectory
        super(WatDensity, self).__init__(trajectory)

        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")

        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.volume = cellpar2volume(self.cellpar)
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements == self.M)[0]
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]

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
        self.d_halfslab = (
            self.xyz[:, -1][self.idx_M].max() - self.xyz[:, -1][self.idx_M].min())/2
        self._idxanchor = 2
        self._transtarget = np.array([0.3, 0.3, -self.d_halfslab])

        # ---------------------------- prepare results array ----------------------------
        # guess the water film thickness
        self.film_thickness = self.get_wat_thickness()
        self.z_bins = np.arange(
            0, self.film_thickness + self.dz_bin, self.dz_bin)
        self.nbins = self.z_bins.shape[0] - 1

        self.hist_oxygen = np.empty(
            (self.n_frames, self.nbins), dtype=np.float32)
        self.hist_hydrogen = np.empty(
            (self.n_frames, self.nbins), dtype=np.float32)

    def _single_frame(self):
        # The calculation
        xyz = self.update_pos(self._ag)
        hist_o, hist_h = self.get_z_density(self._ag)

        # save results
        self.hist_oxygen[self._frame_index] = hist_o
        self.hist_hydrogen[self._frame_index] = hist_h

    def _conclude(self):
        # --------------------- save the original data (frame-wise) ---------------------
        np.save(os.path.join(self.datdir, "hist_oxygen.npy"),   self.hist_oxygen)
        np.save(os.path.join(self.datdir, "hist_hydrogen.npy"), self.hist_hydrogen)

        # ----------------------- calculate time-averaged density -----------------------
        dat_z = self.z_bins[:-1] + self.dz_bin/2
        dV = self.volume*(self.film_thickness/self.cellpar[2]) / self.nbins
        const = self.number_density2watdensity(dV)
        self.hist_oxygen *= const
        self.hist_hydrogen *= const

        dat_oxygen = np.array([dat_z, self.hist_oxygen.mean(axis=0)]).T
        dat_hydrogen = np.array([dat_z, self.hist_hydrogen.mean(axis=0)]).T
        np.savetxt(os.path.join(self.datdir, "oxygen.dat"),
                   dat_oxygen, header="z\tdensity", fmt="%2.6f")
        np.savetxt(os.path.join(self.datdir, "hydrogen.dat"),
                   dat_hydrogen, header="z\tdensity", fmt="%2.6f")
        np.savetxt(os.path.join(self.datdir, "hist2rho_scale.dat"), np.array([const]),
                   header="multiply this constant to histogram density to get number densiity in g/cm^{3} in post-process")

        # -------------------------------- plot density ---------------------------------
        self.plot_density_z(dat_oxygen, dat_hydrogen)

    def get_watOidx(self):
        r"""This methods would select all water oxygen index

        Returns:
            tuple: (<water oxygen index>, <all Hydrogen index>). Note, for rutile-\hkl(110)
            water interface, all hydrogen comes from water.
        """
        xyz = self.xyz
        cn_H = count_cn(xyz[self.idx_O, :],
                        xyz[self.idx_H, :], 1.2, None, self.cellpar)
        cn_M = count_cn(xyz[self.idx_O, :], xyz[self.idx_M,
                        :], self.cutoff, None, self.cellpar)
        watOidx = self.idx_O[(cn_H >= 0) * (cn_M <= 1)]
        watHidx = self.idx_H
        return watOidx, watHidx

    def get_wat_thickness(self):
        trans = self._transtarget - self.xyz[self._idxanchor]
        xyz = apply_PBC(self.xyz+trans, self.cellpar)
        if self.rotM is not None:
            xyz = np.matmul(xyz, self.rotM)
        thickness = xyz[:, -1][self.watOidx].max() - \
            xyz[:, -1][self.watOidx].min()
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
    """The new rutile (110)-water interface dissociation degree analysis method

    Args:
        AnalysisBase (MDAnalysis):

    Usage examples:
    1) Rutile 110 with <1-11> Edge:
        ```python
        from toolkit.structures.rutile110 import Rutile1p11Edge

        atoms = read("init.cif")
        r110edge = Rutile1p11Edge(atoms, vecy=vecy, vecz=vecz, cutoff=2.9)
        owidx, _ = r110edge.get_wat()
        ind      = r110edge.get_indices()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)

        cn5idx = ind['idx_M5c'].reshape(2, -1)
        edge5idx = ind['idx_edge_M5c'].reshape(2, -1)
        edge4idx = ind['idx_edge_M4c'].reshape(2, -1)

        disdeg   = RutileDisDeg(ag, owidx, cn5idx, nrow=r110edge.nrow,
                                edge4idx=edge4idx, edge5idx=edge5idx,)
        disdeg.run()
        ```
    2) Flat model:
        ```python
        atoms = read("init.cif")
        r110  = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
        owidx, _ = r110.get_wat()
        cn5idx   = r110.get_indices()['idx_M5c']
        disdeg   = RutileDisDeg(ag, owidx, cn5idx, nrow=r110.nrow)
        disdeg.run()
        ```
    """

    def __init__(self, atomgroup, owidx, cn5idx, edge4idx=None, edge5idx=None, nrow=2, M='Ti', bins=500, n_oh=2):
        r"""Initializing rutile interface dissociation degree analysis

        Args:
            atomgroup (MDAnalysis.Atomgroup):
                Just use all the atoms in your universe. universe.atoms
            owidx (np.ndarray):
                1-d integer array, which contains the indices for water oxygen in your inteface model.
                you can use medthod 'get_watOidx' to get this index.
            cn5idx (np.ndarray):
                2-d integer array, which contains all the indices for terrace Ti_{5c} atoms in your
                interface model. This can be obtained using Method 'Rutile1p11Edge' or 'Rutile110'.
            edge4idx (np.ndarray, optional):
                2-d integer array, which contains all the indices for edge Ti_{4c} atoms in your
                interface model. This option is specially tailored for \hkl<1-11> edge model. You don't need
                to specify this. Defaults to None.
            edge5idx (np.ndarray, optional):
                2-d integer array, which contains all the indices for edge Ti_{5c} atoms in your
                interface model. This option is specially tailored for \hkl<1-11> edge model. You don't need
                to specify this. Defaults to None.
            M (str, optional):
                Metal element in rutile strucure. Defaults to 'Ti'.
            bins (float, optional):
                Bin size for output O-H distances histogram ouput. Typical OH distances range from 0.85 to 3.5
                angstrom. Defaults to 500 -> typical binsize 0.005 angstrom.
        """

        # load inputs
        self._ag = atomgroup
        self.owidx = owidx
        self.cn5idx = cn5idx.flatten()
        if edge4idx is not None:
            self.edge4idx = edge4idx.flatten()
            self.edge5idx = edge5idx.flatten()
            self.is_step = True
        else:
            self.is_step = False
        self.nrow = nrow
        self.M = M
        self.bins = bins
        self.n_oh = n_oh

        # MDA analysis class routine
        trajectory = atomgroup.universe.trajectory
        super(RutileDisDeg, self).__init__(trajectory)

        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")

        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)

        # the name for output files
        self.fn_distOH5s = os.path.join(self.datdir, "distOH-5s.npy")
        self.fn_distOH5e = os.path.join(self.datdir, "distOH-5e.npy")
        self.fn_distOH4e = os.path.join(self.datdir, "distOH-4e.npy")
        self.fn_disdeg = os.path.join(self.datdir, "disdeg.npy")
        self.fn_histOadH1 = os.path.join(self.datdir, "histOH1.dat")
        self.fn_histOadH2 = os.path.join(self.datdir, "histOH2.dat")

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements == self.M)[0]
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]
        self.bin_edges = np.linspace(0.85, 3.5, self.bins+1)
        self.r = self.bin_edges[:-1] + (self.bin_edges[1]-self.bin_edges[0])/2

        # -------------------------- prepare temporary arraies --------------------------
        self._adO_5s = np.empty(self.cn5idx.shape[0], dtype=int)
        self.dm_cn5 = np.empty(
            (self.cn5idx.shape[0], self.owidx.shape[0]), dtype=float)
        self.dm_O5s = np.empty(
            (self.cn5idx.shape[0], self.idx_H.shape[0]), dtype=float)
        if self.is_step:
            self._adO_5e = np.empty(self.edge5idx.shape[0], dtype=int)
            self._adO_4e = np.empty(self.edge4idx.shape[0]*2, dtype=int)
            self.dm_edge4 = np.empty(
                (self.edge4idx.shape[0], self.owidx.shape[0]), dtype=float)
            self.dm_edge5 = np.empty(
                (self.edge5idx.shape[0], self.owidx.shape[0]), dtype=float)
            self.dm_O5e = np.empty(
                (self.edge5idx.shape[0], self.idx_H.shape[0]), dtype=float)
            self.dm_O4e = np.empty(
                (self.edge4idx.shape[0]*2, self.idx_H.shape[0]), dtype=float)

        # ---------------------------- prepare results array ----------------------------
        self.dist_5s = np.empty(
            (self.n_frames, self.n_oh, self.cn5idx.shape[0]), dtype=np.float32)
        if self.is_step:
            self.dist_5e = np.empty(
                (self.n_frames, 2, self.edge5idx.shape[0]), dtype=np.float32)
            self.dist_4e = np.empty(
                (self.n_frames, 2, self.edge4idx.shape[0]*2), dtype=np.float32)
            self._n = self.cn5idx.shape[0] + \
                self.edge4idx.shape[0] + self.edge5idx.shape[0]
        else:
            self._n = self.cn5idx.shape[0]
        self.disdeg = np.empty((self.n_frames, 2), dtype=np.float32)

    def _single_frame(self):
        self.get_neighbor_oxygen(self.cn5idx,   self.dm_cn5,   self._adO_5s, 1)
        self.dist_5s[self._frame_index, ] = self.get_OH_dist(
            self._adO_5s, self.dm_O5s).T
        if self.is_step:
            self.get_neighbor_oxygen(
                self.edge5idx, self.dm_edge5, self._adO_5e, 1)
            self.get_neighbor_oxygen(
                self.edge4idx, self.dm_edge4, self._adO_4e, 2)
            # Here we sort self._adO4e such that, for all 8 Ti-4c adsorbed waters, we have
            # upper Oad*2, upper Oad-edge*2, lower Oad*2, lower Oad-edge*2.
            # This way, we can use numpy reshape and mean method to get the averages of equivalent sites
            sort_idx = np.argsort(self._ag.positions[self._adO_4e][:, -1])
            upper_tmp = np.flip(
                self._adO_4e[sort_idx][self.edge4idx.shape[0]:])
            lower_tmp = self._adO_4e[sort_idx][:self.edge4idx.shape[0]]
            self._adO_4e = np.array([upper_tmp.reshape(2, -1, order="F"),
                                    lower_tmp.reshape(2, -1, order="F")]).flatten()
            # print(self._adO_4e)
            self.dist_5e[self._frame_index, ] = self.get_OH_dist(
                self._adO_5e, self.dm_O5e).T
            self.dist_4e[self._frame_index, ] = self.get_OH_dist(
                self._adO_4e, self.dm_O4e).T

    def _conclude(self):
        np.save(self.fn_distOH5s, self.dist_5s)
        if self.is_step:
            np.save(self.fn_distOH5e, self.dist_5e)
            np.save(self.fn_distOH4e, self.dist_4e)

        hist1_5s, hist2_5s = self.dist2histo(
            self.dist_5s, self.bin_edges, self.nrow)
        if self.is_step:
            hist1_5e, hist2_5e = self.dist2histo(
                self.dist_5e, self.bin_edges, self.nrow)
            hist1_4e, hist2_4e = self.dist2histo(
                self.dist_4e, self.bin_edges, self.nrow)
            header_list = ["d(Oad-H) [A]"] + ["Oad-half"] + [r"Oad #{0}".format(
                ii) for ii in range(len(hist1_5s)+1)] + ["Oad-edge"]
            hist1 = hist1_5e + hist1_5s + hist1_4e
            hist2 = hist2_5e + hist2_5s + hist2_4e
        else:
            hist1 = hist1_5s
            hist2 = hist2_5s
            header_list = [
                "d(Oad-H) [A]"] + [r"Oad #{0}".format(ii) for ii in range(len(hist1_5s))]

        res1 = np.concatenate([[self.r], hist1], axis=0)
        res2 = np.concatenate([[self.r], hist2], axis=0)
        np.savetxt(self.fn_histOadH1, res1.T, fmt="%10.6f",
                   header="\t".join(header_list))
        np.savetxt(self.fn_histOadH2, res2.T, fmt="%10.6f",
                   header="\t".join(header_list))

        if self.is_step:
            cn_5s = self.dist2cn(self.dist_5s)
            cn_5e = self.dist2cn(self.dist_5e)
            cn_4e = self.dist2cn(self.dist_4e)
            cn = np.concatenate([cn_5s, cn_5e, cn_4e], axis=-1)
        else:
            cn = self.dist2cn(self.dist_5s)
        self.disdeg[:] = self.cn2disdeg(cn)
        np.save(self.fn_disdeg, self.disdeg)

    def get_neighbor_oxygen(self, idx_ti, res_dm, res_idx, n_ow=1):
        """Give a group of ti atoms, find their neighboring water oxygen within cutoff rasius `self.cutoff`.
        Returns water oxygen indices

        Args:
            idx_ti (np.ndarray):
                1-d integer array containing indices for Ti5c atoms.
            res_dm (np.ndarray):
                2-d distance array containing pair-wise distance between Ti and all water oxygen atoms.
                This array is prepared in `_prepare`.
            n_ow (int, optional):
                number of neighboring oxygen for input group of Ti atoms. For example, Ti5c has 1 ad-water;
                and edge Ti4c has 2 ad-water. Defaults to 1.

        Returns:
            np.ndarray:
                1-d array containing adsorbed water oxygen indices. If these is no water oxygen within the
                cutoff raidius, a masking value of '-1' is provided
        """
        # group_idx
        distance_array(
            self._ag.positions[idx_ti], self._ag.positions[self.owidx], result=res_dm, box=self.cellpar)
        sort_idx = np.argsort(res_dm, axis=1)
        res_idx[:] = self.owidx[sort_idx[:, :n_ow]].flatten()

    def get_OH_dist(self, idx_O, res_dm):
        distance_array(self._ag.positions[idx_O], self._ag.positions[self.idx_H],
                       result=res_dm, box=self.cellpar)
        return np.sort(res_dm, axis=1)[:, :self.n_oh]

    @staticmethod
    def dist2histo(dist, bin_edges, nrow):
        n_frames = dist.shape[0]
        dist1 = dist[:, 0].reshape(n_frames, 2, nrow, -1)
        dist2 = dist[:, 1].reshape(n_frames, 2, nrow, -1)
        nTiTerms = dist1.shape[-1]
        dist1, dist2 = dist1.reshape(-1, nTiTerms), dist2.reshape(-1, nTiTerms)
        hist1_list = []
        hist2_list = []
        for ii in range(dist1.shape[-1]):
            hist1, _ = np.histogram(dist1[:, ii], bins=bin_edges, density=True)
            hist2, _ = np.histogram(dist2[:, ii], bins=bin_edges, density=True)
            hist1_list.append(hist1)
            hist2_list.append(hist2)
        return hist1_list, hist2_list

    @staticmethod
    def dist2cn(dist):
        cn = dist[:, 0].astype(np.int8)
        cn[:] = -100
        cn[dist[:, 1] < 1.2] = 2
        cn[dist[:, 0] > 1.2] = 0
        cn[(dist[:, 0] < 1.2) & (dist[:, 1] > 1.2)] = 1
        return cn

    @staticmethod
    def cn2disdeg(cn):
        nframes = cn.shape[0]
        nsite = cn.shape[-1]//2
        cn = cn.reshape(nframes, 2, nsite)
        return 1 - np.sum(cn == 2, axis=-1)/nsite


class staleRutileDisDeg(AnalysisBase):
    # stale rutile dissociation degree analysis.
    # Feature:
    # - use hard cutoff (1.2 A) to determine proton dissociation
    # - will save raw data: coordination number surf-water and surf-water oxygen atom indices
    r"""MDAnalysis class calculating surface water dissociation degree for rutile \hkl(110)-water interface.
    Besides dissociation, this method will also output surface adsorption water oxygen index, which is useful for
    TiO2-water interface, because adsorbed water in this system sometimes exchange with sub-interface water.

    Args:
        AnalysisBase (object): MDAnalysis Analysis class base

    Usage example:
    1. for rutile 110 (water) interface with <1-11> edge
        ```python
        vecy = np.array([26.34844236,  1.8642182,  -2.0615678])
        vecz = np.array([0.49339,  -0.070221,  9.201458])
        r110edge = Rutile1p11Edge(atoms, vecy=vecy, vecz=vecz, M="Ti", nrow=2)
        ind = r110edge.get_indices()
        idx_ow, _ = r110edge.get_wat()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
        cn5idx = ind['idx_M5c'].reshape(2, -1)
        edge5idx = ind['idx_edge_M5c'].reshape(2, -1)
        edge4idx = ind['idx_edge_M4c'].reshape(2, -1)
        rdd = RutileDisDeg(ag, idx_ow, cn5idx, edge4idx=edge4idx, edge5idx=edge5idx)
        rdd.run()
        ```
    """

    def __init__(self, atomgroup, owidx, cn5idx, edge4idx=None, edge5idx=None, M='Ti', cutoff=2.8):
        r"""Initialising a dissociation degree calculating class

        Args:
            atomgroup (MDAnalysis.Atomgroup):
                Just use all the atoms in your universe. universe.atoms
            owidx (np.ndarray):
                1-d integer array, which contains the indices for water oxygen in your inteface model.
                you can use medthod 'get_watOidx' to get this index.
            cn5idx (np.ndarray):
                2-d integer array, which contains all the indices for terrace Ti_{5c} atoms in your
                interface model. This can be obtained using Method 'Rutile1p11Edge' or 'Rutile110'.
            edge4idx (np.ndarray, optional):
                2-d integer array, which contains all the indices for edge Ti_{4c} atoms in your
                interface model. This option is specially tailored for \hkl<1-11> edge model. You don't need
                to specify this. Defaults to None.
            edge5idx (np.ndarray, optional):
                2-d integer array, which contains all the indices for edge Ti_{5c} atoms in your
                interface model. This option is specially tailored for \hkl<1-11> edge model. You don't need
                to specify this. Defaults to None.
            M (str, optional):
                Metal element in rutile strucure. Defaults to 'Ti'.
            cutoff (float, optional):
                cutoff distance for calculating coordination number of M-O. For TiO2 \hkl(110)-water interface,
                2.8 angstrom works well. This value is used for determining adsorbed water above terrace Ti5c
                atoms. Distances between water oxygen and Ti5c lager than `cutoff` will consider as **not**
                adsorbed. Defaults to 2.8 angstrom.
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
        self.fn_adind_cn5 = os.path.join(self.datdir, "ad_O_indices.npy")
        self.fn_adind_edge5 = os.path.join(
            self.datdir, "ad_O_indices-edge5.npy")
        self.fn_adind_edge4 = os.path.join(
            self.datdir, "ad_O_indices-edge4.npy")
        self.fn_cn = os.path.join(self.datdir, "SurfaceOxygenCN.npy")
        self.fn_disdeg5s = os.path.join(self.datdir, "disdeg-Ti5s.npy")
        self.fn_disdeg4e = os.path.join(self.datdir, "disdeg-Ti4e.npy")
        self.fn_disdeg5e = os.path.join(self.datdir, "disdeg-Ti5e.npy")

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements == self.M)[0]
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]

        # -------------------------- prepare temporary arraies --------------------------
        self.dm_cn5 = np.empty(
            (self.n_cn5idx, self.owidx.shape[0]), dtype=float)
        self.dm_edge4 = np.empty(
            (self.n_edge4idx, self.owidx.shape[0]), dtype=float)
        self.dm_edge5 = np.empty(
            (self.n_edge5idx, self.owidx.shape[0]), dtype=float)

        # ---------------------------- prepare results array ----------------------------
        self._n = self.n_cn5idx + self.n_edge4idx*2 + self.n_edge5idx
        self.ad_indices = np.empty((self.n_frames, 2, self._n), dtype=int)
        self.cn = np.empty((self.n_frames, 2, self._n), dtype=float)
        self.disdeg5s = np.empty((self.n_frames, 2, 4), dtype=float)
        self.disdeg4e = np.empty((self.n_frames, 2, 4), dtype=float)
        self.disdeg5e = np.empty((self.n_frames, 2, 4), dtype=float)

    def _single_frame(self):
        # get 'neighbor' (Adsorbed) oxygen indices
        # indices being -1 means no adsorbed oxygen atoms
        # within a cutoff sphere (self.cutoff)
        self.ad_indices[self._frame_index, 0, :self.n_cn5idx] = \
            self.get_neighbor_oxygen(self.upper_cn5idx, self.dm_cn5, n_ow=1)
        self.ad_indices[self._frame_index, 1, :self.n_cn5idx] = \
            self.get_neighbor_oxygen(self.lower_cn5idx, self.dm_cn5, n_ow=1)
        if self.n_edge5idx > 0:
            self.ad_indices[self._frame_index, 0, self.n_cn5idx:(self.n_cn5idx+self.n_edge5idx)] = \
                self.get_neighbor_oxygen(
                    self.upper_edge5idx, self.dm_edge5, n_ow=1)
            self.ad_indices[self._frame_index, 1, self.n_cn5idx:(self.n_cn5idx+self.n_edge5idx)] = \
                self.get_neighbor_oxygen(
                    self.lower_edge5idx, self.dm_edge5, n_ow=1)
        if self.n_edge4idx > 0:
            self.ad_indices[self._frame_index, 0, (self.n_cn5idx+self.n_edge5idx):] = \
                self.get_neighbor_oxygen(
                    self.upper_edge4idx, self.dm_edge4, n_ow=2)
            self.ad_indices[self._frame_index, 1, (self.n_cn5idx+self.n_edge5idx):] = \
                self.get_neighbor_oxygen(
                    self.upper_edge4idx, self.dm_edge4, n_ow=2)

        # calculate the coordination numnber for the oxygen
        idx_o = self.ad_indices[self._frame_index].flatten()
        mask = (idx_o == -1)   # mask using -1 indices
        cn = count_cn(self._ag.positions[idx_o], self._ag.positions[self.idx_H],
                      cutoff_hi=1.2, cutoff_lo=None, cell=self.cellpar).astype(float)
        cn[mask] = np.nan
        self.cn[self._frame_index] = cn.reshape(2, self._n)

    def _conclude(self):
        res_cn5_Oind = self.ad_indices[:, :, :self.n_cn5idx]
        np.save(self.fn_adind_cn5, res_cn5_Oind)
        if self.n_edge5idx > 0:
            res_edge5_Oind = self.ad_indices[:, :, self.n_cn5idx:(
                self.n_cn5idx+self.n_edge5idx)]
            np.save(self.fn_adind_edge5, res_edge5_Oind)
        if self.n_edge4idx > 0:
            res_edge4_Oind = \
                self.ad_indices[:, :, (self.n_cn5idx+self.n_edge5idx):].reshape(self.n_frames,
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
                np.nansum(self.cn == kk, axis=-1)/n_effect
        np.save(self.fn_disdeg, self.disdeg)
        return None

    def get_neighbor_oxygen(self, idx_ti, res_dm, n_ow=1):
        """Give a group of ti atoms, find their neighboring water oxygen within cutoff rasius `self.cutoff`.
        Returns water oxygen indices

        Args:
            idx_ti (np.ndarray):
                1-d integer array containing indices for Ti5c atoms.
            res_dm (np.ndarray):
                2-d distance array containing pair-wise distance between Ti and all water oxygen atoms.
                This array is prepared in `_prepare`.
            n_ow (int, optional):
                number of neighboring oxygen for input group of Ti atoms. For example, Ti5c has 1 ad-water;
                and edge Ti4c has 2 ad-water. Defaults to 1.

        Returns:
            np.ndarray:
                1-d array containing adsorbed water oxygen indices. If these is no water oxygen within the
                cutoff raidius, a masking value of '-1' is provided
        """
        # group_idx
        distance_array(
            self._ag.positions[idx_ti], self._ag.positions[self.owidx], result=res_dm, box=self.cellpar)
        sort_idx = np.argsort(res_dm, axis=1)
        res_idx = self.owidx[sort_idx[:, :n_ow]]
        mask = (np.take_along_axis(
            res_dm, sort_idx[:, :n_ow], axis=1) > self.cutoff)
        res_idx[mask] = -1
        return res_idx.reshape(-1)


class dAdBridge(AnalysisBase):
    """MDAnalysis class calculating distances between adsobed water oxygen atoms and two neighboring bridge
    oxygen atoms. The oxygen water oxygen adsorb on terrace Ti5c atoms.
    Args:
        AnalysisBase (object): MDAnalysis Analysis class base

    Usage example:
    1. for flat rutile 110 water interface
        ```python
        atoms = read(os.path.join("init.cif"))
        r110  = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
        idx_owat, _ = r110.get_wat()
        ind = edge_water.get_indices()
        # split cn5idx and obr_idx to [<upper idx>, <lower idx>]
        idx_cn5     = ind['idx_M5c'].reshape(2, -1)
        idx_obr     = ind['idx_Obr'].reshape(2, -1)
        _, upper_obr = pair_M5c_n_obr(atoms, idx_cn5[0], idx_obr[0])
        _, lower_obr = pair_M5c_n_obr(atoms, idx_cn5[1], idx_obr[1])
        idx_obr = np.array([upper_obr, lower_obr])
        dab = dAdBridge(ag, idx_cn5, idx_obr, idx_owat, idx_adO=None)
        dab.run()
        ```
    2. for rutile 110 water interface with edge along <1-11>
        ```python
        atoms = read(os.path.join("init.cif"))
        vecy = np.array([26.34844236,  1.8642182,  -2.0615678])
        vecz = np.array([0.49339,  -0.070221,  9.201458])
        r110edge = Rutile1p11Edge(atoms, vecy=vecy, vecz=vecz, M="Ti", nrow=2)
        ind = r110edge.get_indices()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
        idx_cn5 = ind['idx_M5c'].reshape(2, -1)
        idx_obr = np.concatenate([ind['idx_Obr'].reshape(2, -1),
                                  ind['idx_hObr_upper'].reshape(2, -1)], axis=1)

        _, upper_obr = pair_M5c_n_obr(atoms, idx_cn5[0], idx_obr[0])
        _, lower_obr = pair_M5c_n_obr(atoms, idx_cn5[1], idx_obr[1])
        idx_obr = np.array([upper_obr, lower_obr])
        dab = dAdBridge(ag, cn5idx, idx_obr)
        dab.run()
        ```
    """

    def __init__(self, atomgroup, idx_cn5, idx_obr, idx_owat, idx_adO=None, ref_vec=None, M='Ti', cutoff=2.8):
        """Initialize analysis method 'dAdBridge'

        Args:
            atomgroup (MDAnalysis.Atomgroup):
                Just use all the atoms in your universe. universe.atoms
            idx_cn5 (np.ndarray):
                2-d integer array, with shape (2, n_cn5), which contains the upper and lower indices for
                terrace Ti_{5c} atoms in your interface model. This can be obtained using Method 'Rutile1p11Edge'
                or 'Rutile110'.
            idx_adO (np.ndarray):
                3-d integer array, with shape (n_frames, 2, n_cn5), which contains the indices of adsorbed
                water oxygen atoms at every MD snapshot. This array is obtained using MDAnalsis analysis
                class: RutileDisDeg.
            idx_obr (np.ndarray):
                3-d integer array, with shape (2, n_cn5, 2), which contains the upper and lower indices for
                two rows of bridge oxygen indices corresponding to `idx_cn5`. This can be obtained using Method
                'pair_M5c_n_obr'.
            ref_vec (np.ndarray, optional):
                reference vector v_r. The horizontal distances between adsorption oxygen and pairing bridge oxygen
                is obtained by dot product
                                                np.dot(v_r, v(Oad-Obr)).
                If `ref_vec` is not provided, the normalized vector of averaged v(Obr1-Obr2) will be used.
                Defaults to None.
            M (str, optional):
                Metal element in rutile strucure. Defaults to 'Ti'.. Defaults to 'Ti'.
        """

        # load inputs
        self._ag = atomgroup
        self.idx_cn5 = idx_cn5.flatten()
        self.idx_obr = idx_obr
        self.idx_obr1 = np.array(
            [idx_obr[0, :, 0], idx_obr[1, :, 0]]).flatten()
        self.idx_obr2 = np.array(
            [idx_obr[0, :, 1], idx_obr[1, :, 1]]).flatten()
        self.idx_owat = idx_owat

        self._find_Oad = True
        if idx_adO is not None:
            self.idx_adO = idx_adO
            if len(idx_adO.shape) == 3:
                nf = idx_adO.shape[0]
                self.idx_adO = idx_adO.reshape(nf, -1)
            self._find_Oad = False

        if ref_vec is None:
            self.ref_vec = None
            print("'ref_vec' is not provided. Use auto generated reference vectors.")
        else:
            self.ref_vec = ref_vec
        self.M = M
        self.cutoff = cutoff

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
        self.fn_adind_cn5 = os.path.join(self.datdir, "ad_O_indices.npy")

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements == self.M)[0]
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]
        if self.ref_vec is None:
            ref_upper = self.get_ref(
                self.idx_obr[0][:, 0], self.idx_obr[0][:, 1])
            ref_lower = self.get_ref(
                self.idx_obr[1][:, 0], self.idx_obr[1][:, 1])
            self.ref_vec = (ref_upper + ref_lower)/2

        # ---------------------------- prepare results array ----------------------------
        self.dab = np.empty(
            (self.n_frames, 2, self.idx_cn5.shape[-1]), dtype=float)
        if self._find_Oad is True:
            self.dm_cn5 = np.empty(
                (self.idx_cn5.shape[-1], self.idx_owat.shape[0]), dtype=float)
            self.idx_adO = np.empty(
                (self.n_frames, self.idx_cn5.shape[-1]), dtype=int)

    def _single_frame(self):
        if self._find_Oad:
            self.idx_adO[self._frame_index] = self.get_neighbor_oxygen(
                self.idx_cn5, self.dm_cn5)
        v1, v2 = self.get_dab(self.idx_adO[self._frame_index, :],
                              self.idx_obr1, self.idx_obr2)
        self.dab[self._frame_index, :, ] = np.array([v1, v2])

    def _conclude(self):
        self.dab = self.dab.reshape(
            self.n_frames, 2, 2, self.idx_cn5.shape[-1]//2)
        self.dab = np.concatenate([self.dab[:, 0].reshape(self.n_frames, 2, 1, self.idx_cn5.shape[-1]//2),
                                   self.dab[:, 1].reshape(self.n_frames, 2, 1, self.idx_cn5.shape[-1]//2)],
                                  axis=2)
        self.dab = np.abs(self.dab)
        self.idx_adO = self.idx_adO.reshape(
            self.n_frames, 2, self.idx_cn5.shape[-1]//2)
        np.save(self.fn_upperdab, self.dab[:, 0, :, :])
        np.save(self.fn_lowerdab, self.dab[:, 1, :, :])
        np.save(self.fn_adind_cn5, self.idx_adO)

    def get_ref(self, idx1, idx2):
        """use minimum image vector between two rows of obr as reference vectors"""
        xyz = self._ag.positions
        v_mic = minimize_vectors(xyz[idx1] - xyz[idx2], box=self.cellpar)
        return np.mean(v_mic/np.linalg.norm(v_mic, axis=1)[:, np.newaxis], axis=0)

    def get_dab(self, idx_adO, idx_obr1, idx_obr2):
        """get distances between Adsorption water oxygen and Bridge oxygen, aka, dAB.

        Args:
            idx_adO (np.ndarray):
                indices of ad water oxygen atoms. Use -1 as masking marker, meaning didn't found Obr
                whthin cutoff. (see analysis class `RutileDisDeg`)
            idx_obr1 (np.ndarray):
                indices of row#1 of bridge oxygen atoms.
            idx_obr2 (np.ndarray):
                indices of row#2 of bridge oxygen atoms.

        Returns:
            tuple (np.ndarray, np.ndarray):
                (<result distances: Oad-Obr#1>, <result distances: Oad-Obr#2>)
        """
        # Remember that idx_adO being -1 just means this Oad is not found
        # within a cutoff sphere. Therefore, first prepare mask, then
        # substituting the masked value with 'np.nan'
        mask = (idx_adO == -1)
        xyz = self._ag.positions

        micv1 = minimize_vectors(
            xyz[idx_adO] - xyz[idx_obr1], box=self.cellpar)
        micv2 = minimize_vectors(
            xyz[idx_adO] - xyz[idx_obr2], box=self.cellpar)

        res1 = np.matmul(micv1, self.ref_vec)
        res2 = np.matmul(micv2, self.ref_vec)
        res1[mask] = np.nan
        res2[mask] = np.nan
        return res1, res2

    def get_neighbor_oxygen(self, idx_ti, res_dm, n_ow=1):
        """Give a group of ti atoms, find their neighboring water oxygen within cutoff rasius `self.cutoff`.
        Returns water oxygen indices

        Args:
            idx_ti (np.ndarray):
                1-d integer array containing indices for Ti5c atoms.
            res_dm (np.ndarray):
                2-d distance array containing pair-wise distance between Ti and all water oxygen atoms.
                This array is prepared in `_prepare`.
            n_ow (int, optional):
                number of neighboring oxygen for input group of Ti atoms. For example, Ti5c has 1 ad-water;
                and edge Ti4c has 2 ad-water. Defaults to 1.

        Returns:
            np.ndarray:
                1-d array containing adsorbed water oxygen indices. If these is no water oxygen within the
                cutoff raidius, a masking value of '-1' is provided
        """
        # group_idx
        distance_array(
            self._ag.positions[idx_ti], self._ag.positions[self.idx_owat], result=res_dm, box=self.cellpar)
        sort_idx = np.argsort(res_dm, axis=1)
        res_idx = self.idx_owat[sort_idx[:, :n_ow]]
        mask = (np.take_along_axis(
            res_dm, sort_idx[:, :n_ow], axis=1) > self.cutoff)
        res_idx[mask] = -1
        return res_idx.reshape(-1)


class dInterLayer(AnalysisBase):
    """MDAnalysis class for interlayer distances calculation. Note that this utility is only good
    for flat TiO2 (110)-water interface
    Notice: Because the rotation matrix 'get_rotM' method is not very robust, this interlayer distances
    method is only for flat rutile 110 water interface currently.

    Usage example:
    1. for flat rutile 110 water interface:
        ```python
        atoms  = read(os.path.join("init.cif"))
        r110   = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
        n_ti5c = r110.get_indices()['idx_M5c'].flatten().shape[0]//2
        dil    = dInterLayer(ag, n_ti5c)
        dil.run()
        ```
    """

    def __init__(self, atomgroup, n_ti5c, dz=0.005):
        """ Initializing interlayer distances calculation

        Args:
            atomgroup (MDAnalysis.Atomgroup):
                Just use all the atoms in your universe. universe.atoms
            n_ti5c (int):
                Number of Ti5c's, which is also half of Ti numbers per layer
            dz (float, optional):
                Bin size for inter-layer distances histogram. Defaults to 0.005 angstrom.
        """

        self._ag = atomgroup
        self.idx_Ti = np.where(self._ag.elements == "Ti")[0]
        self.n_ti = 2*n_ti5c
        n = self.idx_Ti.shape[0] // self.n_ti
        z_ti = self._ag.positions[self.idx_Ti][:, -1]
        args = np.argsort(z_ti)
        idx_Ti_layered = self.idx_Ti[args].reshape(n, self.n_ti)
        self.idx_Ti_layered = idx_Ti_layered
        self.n = idx_Ti_layered.shape[0]
        self.n_ti = idx_Ti_layered.shape[1]
        self.dz = dz

        trajectory = atomgroup.universe.trajectory
        super(dInterLayer, self).__init__(trajectory)

        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")

        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)

        self.fn_z = os.path.join(self.datdir, "ti_z_mean.npy")
        self.fn_histo = os.path.join(self.datdir, "ti_z_mean_histo.dat")

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]

        self._idxanchor = 2
        self._transtarget = np.array([1, 1, 3.5])

        xyz = self.update_pos(self.xyz, self._transtarget, self._idxanchor)
        xyz = apply_PBC(xyz, box=self.cellpar)

        _, self.z_min, self.z_max = self.get_n_trilayers(xyz, self.idx_Ti)
        self.bin_edges = np.arange(self.z_min-1, self.z_max+1+self.dz, self.dz)

        # ---------------------------- prepare results array ----------------------------
        self.z_mean = np.empty(
            (self.n_frames, self.idx_Ti_layered.shape[0]), dtype=np.float32)

    def _single_frame(self):
        # update positions
        xyz = self.update_pos(self._ag.positions,
                              self._transtarget, self._idxanchor)
        xyz = apply_PBC(xyz, box=self.cellpar)
        self.z_mean[self._frame_index, :] = xyz[:, -
                                                1][self.idx_Ti_layered].mean(axis=-1)

    def _conclude(self):
        np.save(self.fn_z, self.z_mean)
        self.z = self.bin_edges[1:] - self.dz/2
        hist = self.get_z_histo(self.z_mean.flatten(), self.bin_edges)
        res = np.array([self.z, hist]).T
        np.savetxt(self.fn_histo, res, fmt="%10.6f",
                   header="Z [A]\t\thistogram")

    @staticmethod
    def update_pos(xyz, anchor_pos, anchor_idx):
        """To avoding collective slab drifting during the simulation. Move atom with index 'self._idxanchor'
        to the same position each frame. Defaults to 1 (One of the slab atom if you are using ase for modle
        generation).
        """
        trans = anchor_pos - xyz[anchor_idx]
        xyz += trans
        return xyz

    @staticmethod
    def get_n_trilayers(xyz, idx_Ti):
        EXP_D = 3.3
        z_min = xyz[idx_Ti][:, -1].min()
        z_max = xyz[idx_Ti][:, -1].max()
        n = round((z_max - z_min)/EXP_D) + 1
        return n, z_min, z_max

    @staticmethod
    def get_z_histo(zmean, bin_edges):
        hist, _ = np.histogram(zmean.flatten(), bins=bin_edges, density=True)
        return hist


class SurfTiOBondLenght(AnalysisBase):
    """This class calculates the surface TiO bond length.

    Args:
        AnalysisBase (MDA): MDAnalysis AnalysisBase

    Usage example:
    1. for flat rutile 110 water interface:
        ```python
        atoms = read(os.path.join("init.cif"))
        r110  = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
        idx_owat, _ = r110.get_wat()
        ind = inp.r110.get_indices()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
        ind['idx_Obr'][0] = np.flip(ind['idx_M5c'][0], axis=1)
        idx_cn5  = ind['idx_M5c'].reshape(2, -1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)
        sbl = SurfTiOBondLenght(ag, idx_cn5, idx_obr, idx_owat)
        sbl.run()
        ```
    """

    def __init__(self, atomgroup, idx_cn5, idx_obr, idx_ow, M='Ti'):
        """Initializing surface bond calculation method

        Args:
            atomgroup (MDAnalysis.Atomgroup):
                Just use all the atoms in your universe. universe.atoms
            idx_cn5 (numpy.ndarray):
                Index array for 5-coordinated Surface Ti. The index array should be sorted st. [<upper surface indices>,
                <lower surface indices>]
            idx_obr (numpy.ndarray):
                Index array for Surface bridge O atoms. The index array should be sorted st. [<upper surface indices>,
                <lower surface indices>]
            idx_ow (numpy.ndarray):
                1-d index array for water indices.
            M (str, optional):
                Metal element in rutile strucure. Defaults to 'Ti'.. Defaults to 'Ti'.

        """

        # load inputs
        self._ag = atomgroup
        self.idx_cn5 = idx_cn5.flatten()
        self.idx_obr = idx_obr.flatten()
        self.idx_ow = idx_ow
        self.M = M

        # MDA analysis class routine
        trajectory = atomgroup.universe.trajectory
        super(SurfTiOBondLenght, self).__init__(trajectory)

        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")

        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)

        # the name for output files
        self.fn_dist_tioad = os.path.join(self.datdir, "d_TiOad.npy")
        self.fn_dist_tiobr = os.path.join(self.datdir, "d_TiObr.npy")
        self.fn_indices = os.path.join(self.datdir, "indices.dat")
        # header for output indices
        self.ind_header = "\t\t".join(["Ti5s", "Obr", "Tibr1", "Tibr2"])

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements == self.M)[0]
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]
        self._idx_obr, self._idx_ti = self.pair_TiObr(
            self.xyz, self.idx_obr, self.idx_M, self.cellpar)
        self.indices = np.concatenate(
            [[self.idx_cn5], [self.idx_obr], self._idx_ti.reshape(2, -1)], axis=0)

        # -------------------------- prepare temporary arraies --------------------------
        self.dM_TiOad = np.empty(
            (self.idx_cn5.shape[0], self.idx_ow.shape[0]), dtype=float)
        self.bl_TiObr = np.empty((self._idx_obr.shape[0]), dtype=float)

        # ---------------------------- prepare results array ----------------------------
        self._n = self.idx_cn5.shape[0]
        self.dTiOad = np.empty((self.n_frames, self._n), dtype=float)
        self.dTiObr = np.empty((self.n_frames, self._n*2), dtype=float)

    def _single_frame(self):
        self.dTiOad[self._frame_index, :] = self.get_dTiOad()
        self.dTiObr[self._frame_index, :] = self.get_dTiObr()

    def _conclude(self):
        self.dTiOad = self.dTiOad.reshape(self.n_frames, 2, -1)
        self.dTiObr = self.dTiObr.reshape(self.n_frames, 2, -1)
        np.save(self.fn_dist_tioad, self.dTiOad)
        np.save(self.fn_dist_tiobr, self.dTiObr)
        np.savetxt(self.fn_indices, self.indices.T,
                   fmt="%8d", header=self.ind_header)

    def get_dTiOad(self):
        distance_array(self._ag.positions[self.idx_cn5],
                       self._ag.positions[self.idx_ow],
                       box=self.cellpar, result=self.dM_TiOad)
        return np.min(self.dM_TiOad, axis=1)

    def get_dTiObr(self):
        calc_bonds(self._ag.positions[self._idx_obr],
                   self._ag.positions[self._idx_ti],
                   box=self.cellpar, result=self.bl_TiObr)
        return self.bl_TiObr.copy()

    @staticmethod
    def pair_TiObr(xyz, idx_obr, idx_ti, box):
        xyz1 = xyz[idx_obr]
        xyz2 = xyz[idx_ti]
        dm = distance_array(xyz1, xyz2, box=box)
        sort = np.argsort(dm, axis=1)[:, :2]
        res_obr = np.append(idx_obr, idx_obr)
        res_ti = idx_ti[sort].T.flatten()
        return res_obr, res_ti


class dObr_NearestH(AnalysisBase):
    """Distance between Obr and it's nearest proton.

    Args:
        AnalysisBase (MDAnalysis): MDAnalysis analysis base

    Usage example:
    (1) (110)-water interface with <1-11> edge
        ```python
        from toolkit.structures.rutile110 import Rutile1p11Edge

        atoms = read("init.cif")
        r110edge = Rutile1p11Edge(atoms, vecy=vecy, vecz=vecz, cutoff=2.9)
        idx_owat, _ = r110edge.get_wat()
        ind         = r110edge.get_indices()
        ind['idx_Obr'][0] = np.flip(ind['idx_Obr'][0], axis=1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)
        idx_hobr1 = ind['idx_hObr_mid'].reshape(2, -1)
        idx_hobr2 = ind['idx_hObr_upper'].reshape(2, -1)
        idx_eobr  = ind['idx_edge_O2'].reshape(2, -1)
        doh = dObr_NearestH(ag, idx_obr, nrow=r110edge.nrow, idx_hobr1=idx_hobr1,
                            idx_hobr2=idx_hobr2, idx_eobr=idx_eobr)
        doh.run()
        ```
    (2) Flat (110)-water interface
        ```python
        from toolkit.structures.rutile110 import Rutile110

        atoms    = read("init.cif")
        r110     = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
        idx_owat, _ = r110.get_wat()
        ind = inp.r110.get_indices()
        ind['idx_Obr'][0] = np.flip(ind['idx_Obr'][0], axis=1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)
        doh = dObr_NearestH(ag, idx_obr, nrow=r110.nrow, idx_hobr1=None, idx_hobr2=None, idx_eobr=None)
        doh.run()
        ```
    """

    def __init__(self, atomgroup, idx_obr, nrow=2, idx_hobr1=None, idx_hobr2=None, idx_eobr=None, M='Ti', bins=500):
        """Initializing analysis method

        Args:
            atomgroup (MDAnalysis.Atomgroup):
                Just use all the atoms in your universe. universe.atoms
            idx_obr (numpy.ndarray):
                Index array for Surface bridge O atoms. The index array should be sorted st. [<upper surface indices>,
                <lower surface indices>]
            idx_hobr1 (numpy.ndarray, optional):
                Use this for <1-11> edge. Index array for surface half-Obr atoms. Defaults to None.
            idx_hobr2 (numpy.ndarray, optional):
                Use this for <1-11> edge. Index array for the last Obr, or the other half-Obr. Defaults to None.
            idx_eobr (numpy.ndarray, optional):
                Use this for <1-11> edge. Index array for Obr-edge, 2-coordinated O between Ti-edge4 and Ti-edge5. Defaults to None.
            M (str, optional):
                Metal element in rutile strucure. Defaults to 'Ti'.
            bins (float, optional):
                Bin size for output O-H distances histogram ouput. Typical OH distances range from 0.85 to 3.5
                angstrom. Defaults to 500 -> typical binsize 0.005 angstrom.
        """

        self._ag = atomgroup
        self.M = M
        self.bins = bins
        self.nrow = nrow

        # initializing plane_idx_array
        self.idx_obr = idx_obr.flatten()
        self.is_flat = True
        if idx_hobr1 is not None:
            self.is_flat = False
            self.idx_hobr1 = idx_hobr1.flatten()
            self.idx_hobr2 = idx_hobr2.flatten()
            self.idx_eobr = idx_eobr.flatten()
            self.idx_total_obr = np.concatenate([self.idx_obr,
                                                 self.idx_hobr1,
                                                 self.idx_hobr2,
                                                 self.idx_eobr], axis=-1)
        else:
            self.idx_total_obr = self.idx_obr

        # MDA analysis class routine
        trajectory = atomgroup.universe.trajectory
        super(dObr_NearestH, self).__init__(trajectory)

        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")

        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)

        # the file names for output data
        self.fn_obr_h = os.path.join(self.datdir, "d_Obr-H.npy")
        self.fn_histObrH = os.path.join(self.datdir, "histObrH.dat")
        if not self.is_flat:
            self.fn_hobr1_h = os.path.join(self.datdir, "d_hObr1-H.npy")
            self.fn_hobr2_h = os.path.join(self.datdir, "d_hObr2-H.npy")
            self.fn_eobr_h = os.path.join(self.datdir, "d_eObr-H.npy")

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements == self.M)[0]
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]
        self.bin_edges = np.linspace(0.85, 3.5, self.bins+1)
        self.r = self.bin_edges[:-1] + (self.bin_edges[1]-self.bin_edges[0])/2
        self.n_obr = self.idx_obr.shape[-1]
        if not self.is_flat:
            self.n_hobr1 = self.idx_hobr1.shape[-1]
            self.n_hobr2 = self.idx_hobr2.shape[-1]
            self.n_eobr = self.idx_eobr.shape[-1]
            self.n_total_obr = self.n_obr + self.n_hobr1 + self.n_hobr2 + self.n_eobr
        else:
            self.n_total_obr = self.n_obr

        # -------------------------- prepare temporary arraies --------------------------
        self._dmatrix = np.empty(
            (self.n_total_obr, self.idx_H.shape[0]), dtype=float)

        # ---------------------------- prepare results array ----------------------------
        self.distances = np.empty(
            (self.n_frames, self.n_total_obr), dtype=float)

    def _single_frame(self):
        self.distances[self._frame_index,
                       :] = self.get_min_OH(self.idx_total_obr)

    def _conclude(self):
        self.distances = self.distances.reshape(self.n_frames, 2,
                                                self.n_total_obr//2)
        self.dist_obr_h = self.distances[:, :, :self.n_obr//2]
        hist_obr = self.dist2histo(self.dist_obr_h, self.bin_edges, self.nrow)
        np.save(self.fn_obr_h, self.dist_obr_h)
        if not self.is_flat:
            self.dist_hobr1_h = self.distances[:, :,
                                               self.n_obr//2:(self.n_obr+self.n_hobr1)//2]
            hist_obr_n1 = self.dist2histo(
                self.dist_hobr1_h, self.bin_edges, self.nrow)
            self.dist_hobr2_h = self.distances[:, :, (self.n_obr+self.n_hobr1)//2:(
                self.n_obr+self.n_hobr1+self.n_hobr2)//2]
            hist_hobr = self.dist2histo(
                self.dist_hobr2_h, self.bin_edges, self.nrow)
            self.dist_eobr_h = self.distances[:, :,
                                              (self.n_obr+self.n_hobr1+self.n_hobr2)//2:]
            hist_e2 = self.dist2histo(
                self.dist_eobr_h, self.bin_edges, self.nrow)
            np.save(self.fn_hobr1_h, self.dist_hobr1_h)
            np.save(self.fn_hobr2_h, self.dist_hobr2_h)
            np.save(self.fn_eobr_h, self.dist_eobr_h)
            hist_list = hist_hobr + hist_obr + hist_obr_n1 + hist_e2
            header = "\t".join(['bin edges [A]'] + [r"O_br-half"] +
                                                   [r"O_br#%d" % (ii) for ii in range(self.n_obr//self.nrow//2)] +
                                                   [r"O_br#%d" % (self.n_obr//self.nrow//2)] +
                                                   [r"O_br-edge"])
        else:
            header = "\t".join(['bin edges [A]'] + [r"O_br#%d" % (ii)
                               for ii in range(self.n_obr//self.nrow//2)])
            hist_list = hist_obr

        hist_dat = np.concatenate([[self.r], hist_list], axis=0)
        np.savetxt(self.fn_histObrH, hist_dat.T, fmt="%10.6f", header=header)

    def get_min_OH(self, obr_indices):
        xyz = self._ag.positions
        obr_pos = xyz[obr_indices, :]
        h_pos = xyz[self.idx_H, :]
        distance_array(obr_pos, h_pos, box=self.cellpar, result=self._dmatrix)
        d_oh_min = self._dmatrix.min(axis=1)
        # sel_h_idx = self.idx_H[np.argmin(self._dmatrix, axis=1)]
        # return d_oh_min, sel_h_idx
        return d_oh_min

    @staticmethod
    def dist2histo(dist, bin_edges, nrow):
        n_frames = dist.shape[0]
        dist = dist.reshape(n_frames, 2, nrow, -1)
        nO2Terms = dist.shape[-1]
        dist = dist.reshape(-1, nO2Terms)
        hist_list = []
        for ii in range(dist.shape[-1]):
            hist, _ = np.histogram(dist[:, ii], bins=bin_edges, density=True)
            hist_list.append(hist)
        return hist_list


class dObr_NearH(AnalysisBase):
    """Distance between Obr and it's near n_oh proton.

    Args:
        AnalysisBase (MDAnalysis): MDAnalysis analysis base

    Usage example:
    (1) (110)-water interface with <1-11> edge
        ```python
        from toolkit.structures.rutile110 import Rutile1p11Edge

        atoms = read("init.cif")
        r110edge = Rutile1p11Edge(atoms, vecy=vecy, vecz=vecz, cutoff=2.9)
        idx_owat, _ = r110edge.get_wat()
        ind         = r110edge.get_indices()
        ind['idx_Obr'][0] = np.flip(ind['idx_Obr'][0], axis=1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)
        idx_hobr1 = ind['idx_hObr_mid'].reshape(2, -1)
        idx_hobr2 = ind['idx_hObr_upper'].reshape(2, -1)
        idx_eobr  = ind['idx_edge_O2'].reshape(2, -1)
        doh = dObr_NearestH(ag, idx_obr, nrow=r110edge.nrow, idx_hobr1=idx_hobr1,
                            idx_hobr2=idx_hobr2, idx_eobr=idx_eobr)
        doh.run()
        ```
    (2) Flat (110)-water interface
        ```python
        from toolkit.structures.rutile110 import Rutile110

        atoms    = read("init.cif")
        r110     = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
        idx_owat, _ = r110.get_wat()
        ind = inp.r110.get_indices()
        ind['idx_Obr'][0] = np.flip(ind['idx_Obr'][0], axis=1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)
        doh = dObr_NearestH(ag, idx_obr, nrow=r110.nrow, idx_hobr1=None, idx_hobr2=None, idx_eobr=None)
        doh.run()
        ```
    """

    def __init__(self, atomgroup, idx_obr, nrow=2, idx_hobr1=None, idx_hobr2=None, idx_eobr=None, M='Ti', bins=500, n_oh=5):
        """Initializing analysis method

        Args:
            atomgroup (MDAnalysis.Atomgroup):
                Just use all the atoms in your universe. universe.atoms
            idx_obr (numpy.ndarray):
                Index array for Surface bridge O atoms. The index array should be sorted st. [<upper surface indices>,
                <lower surface indices>]
            idx_hobr1 (numpy.ndarray, optional):
                Use this for <1-11> edge. Index array for surface half-Obr atoms. Defaults to None.
            idx_hobr2 (numpy.ndarray, optional):
                Use this for <1-11> edge. Index array for the last Obr, or the other half-Obr. Defaults to None.
            idx_eobr (numpy.ndarray, optional):
                Use this for <1-11> edge. Index array for Obr-edge, 2-coordinated O between Ti-edge4 and Ti-edge5. Defaults to None.
            M (str, optional):
                Metal element in rutile strucure. Defaults to 'Ti'.
            bins (float, optional):
                Bin size for output O-H distances histogram ouput. Typical OH distances range from 0.85 to 3.5
                angstrom. Defaults to 500 -> typical binsize 0.005 angstrom.
        """

        self._ag = atomgroup
        self.M = M
        self.bins = bins
        self.nrow = nrow
        self.n_oh = n_oh

        # initializing plane_idx_array
        self.idx_obr = idx_obr.flatten()
        self.is_flat = True
        if idx_hobr1 is not None:
            self.is_flat = False
            self.idx_hobr1 = idx_hobr1.flatten()
            self.idx_hobr2 = idx_hobr2.flatten()
            self.idx_eobr = idx_eobr.flatten()
            self.idx_total_obr = np.concatenate([self.idx_obr,
                                                 self.idx_hobr1,
                                                 self.idx_hobr2,
                                                 self.idx_eobr], axis=-1)
        else:
            self.idx_total_obr = self.idx_obr

        # MDA analysis class routine
        trajectory = atomgroup.universe.trajectory
        super(dObr_NearH, self).__init__(trajectory)

        # make/backup data directories
        self.datdir = os.path.join(".", "data_output")
        self.figdir = os.path.join(".", "figure_output")

        create_path(self.datdir, bk=False)
        create_path(self.figdir, bk=False)

        # the file names for output data
        self.fn_obr_h = os.path.join(self.datdir, "d_Obr-H.npy")
        self.fn_histObrH = os.path.join(self.datdir, "histObrH.dat")
        if not self.is_flat:
            self.fn_hobr1_h = os.path.join(self.datdir, "d_hObr1-H.npy")
            self.fn_hobr2_h = os.path.join(self.datdir, "d_hObr2-H.npy")
            self.fn_eobr_h = os.path.join(self.datdir, "d_eObr-H.npy")

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.idx_M = np.where(self._ag.elements == self.M)[0]
        self.idx_O = np.where(self._ag.elements == 'O')[0]
        self.idx_H = np.where(self._ag.elements == 'H')[0]
        self.bin_edges = np.linspace(0.85, 3.5, self.bins+1)
        self.r = self.bin_edges[:-1] + (self.bin_edges[1]-self.bin_edges[0])/2
        self.n_obr = self.idx_obr.shape[-1]
        if not self.is_flat:
            self.n_hobr1 = self.idx_hobr1.shape[-1]
            self.n_hobr2 = self.idx_hobr2.shape[-1]
            self.n_eobr = self.idx_eobr.shape[-1]
            self.n_total_obr = self.n_obr + self.n_hobr1 + self.n_hobr2 + self.n_eobr
        else:
            self.n_total_obr = self.n_obr

        # -------------------------- prepare temporary arraies --------------------------
        self._dmatrix = np.empty(
            (self.n_total_obr, self.idx_H.shape[0]), dtype=float)

        # ---------------------------- prepare results array ----------------------------
        self.distances = np.empty(
            (self.n_frames, self.n_total_obr, self.n_oh), dtype=float)

    def _single_frame(self):
        self.distances[self._frame_index,
                       :] = self.get_OH_dist(self.idx_total_obr)

    def _conclude(self):
        self.distances = self.distances.reshape(self.n_frames, 2,
                                                self.n_total_obr//2, self.n_oh)
        self.dist_obr_h = self.distances[:, :, :self.n_obr//2, :]
        hist_obr = self.dist2histo(
            self.dist_obr_h[:, :, :, 0], self.bin_edges, self.nrow)
        np.save(self.fn_obr_h, self.dist_obr_h)
        if not self.is_flat:
            self.dist_hobr1_h = self.distances[:, :,
                                               self.n_obr//2:(self.n_obr+self.n_hobr1)//2, 0]
            hist_obr_n1 = self.dist2histo(
                self.dist_hobr1_h, self.bin_edges, self.nrow)
            self.dist_hobr2_h = self.distances[:, :, (self.n_obr+self.n_hobr1)//2:(
                self.n_obr+self.n_hobr1+self.n_hobr2)//2, 0]
            hist_hobr = self.dist2histo(
                self.dist_hobr2_h, self.bin_edges, self.nrow)
            self.dist_eobr_h = self.distances[:, :,
                                              (self.n_obr+self.n_hobr1+self.n_hobr2)//2:, 0]
            hist_e2 = self.dist2histo(
                self.dist_eobr_h, self.bin_edges, self.nrow)
            np.save(self.fn_hobr1_h, self.dist_hobr1_h)
            np.save(self.fn_hobr2_h, self.dist_hobr2_h)
            np.save(self.fn_eobr_h, self.dist_eobr_h)
            hist_list = hist_hobr + hist_obr + hist_obr_n1 + hist_e2
            header = "\t".join(['bin edges [A]'] + [r"O_br-half"] +
                                                   [r"O_br#%d" % (ii) for ii in range(self.n_obr//self.nrow//2)] +
                                                   [r"O_br#%d" % (self.n_obr//self.nrow//2)] +
                                                   [r"O_br-edge"])
        else:
            header = "\t".join(['bin edges [A]'] + [r"O_br#%d" % (ii)
                               for ii in range(self.n_obr//self.nrow//2)])
            hist_list = hist_obr

        hist_dat = np.concatenate([[self.r], hist_list], axis=0)
        np.savetxt(self.fn_histObrH, hist_dat.T, fmt="%10.6f", header=header)

    def get_OH_dist(self, obr_indices):
        xyz = self._ag.positions
        obr_pos = xyz[obr_indices, :]
        h_pos = xyz[self.idx_H, :]
        distance_array(obr_pos, h_pos, result=self._dmatrix, box=self.cellpar)
        return np.sort(self._dmatrix, axis=1)[:, :self.n_oh]

    @staticmethod
    def dist2histo(dist, bin_edges, nrow):
        n_frames = dist.shape[0]
        dist = dist.reshape(n_frames, 2, nrow, -1)
        nO2Terms = dist.shape[-1]
        dist = dist.reshape(-1, nO2Terms)
        hist_list = []
        for ii in range(dist.shape[-1]):
            hist, _ = np.histogram(dist[:, ii], bins=bin_edges, density=True)
            hist_list.append(hist)
        return hist_list


class FindSurfaceOadH(AnalysisBase):
    """_summary_
    Not applicable to Edge models
    Usage Examples:
        1) Flat model:
        ```python
        atoms = read("init.cif")
        r110  = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
        Ow_idx, _ = r110.get_wat()
        Ti5c_idx  = r110.indices["idx_M5c"].flatten()
        findOH   = FindSurfaceOadH(ag, owidx, cn5idx, nrow=r110.nrow)
        findOH.run()
        ```
    Args:
        AnalysisBase (_type_): _description
    """

    def __init__(self, atomgroup, Ow_idx, M5c_idx, M='Ti'):
        self._ag = atomgroup
        self.M = M
        self.Ow_idx = Ow_idx
        self.M5c_idx = M5c_idx

        # MDA analysis class routine
        trajectory = atomgroup.universe.trajectory
        super(FindSurfaceOadH, self).__init__(trajectory)

        # make/backup data directories
        self.datdir = os.path.join(".", "SurfaceOadH")

        create_path(self.datdir, bk=False)

        # the file names for output data
        self.fn_all_info = os.path.join(self.datdir, "surf_TiOH_cprs.npz")

    def _prepare(self):
        # ------------------------ initialize usefule constants -------------------------
        self.cellpar = self._ag.dimensions
        self.xyz = self._ag.positions
        self.M_idx = np.where(self._ag.elements == self.M)[0]
        self.O_idx = np.where(self._ag.elements == 'O')[0]
        self.H_idx = np.where(self._ag.elements == 'H')[0]

        self.num_M5c = len(self.M5c_idx)

        # -------------------------- prepare result arrays --------------------------
        self.all_Oad_H_idx = np.full((self.n_frames, self.num_M5c), -1)
        self.all_M5c_OadH_idx = np.full((self.n_frames, self.num_M5c), -1)
        self.all_Had_idx = np.full((self.n_frames, self.num_M5c), -1)
        self.all_bond_O_H = np.full((self.n_frames, self.num_M5c), np.nan)
        self.all_angle_M_O_H = np.full((self.n_frames, self.num_M5c), np.nan)

    def _single_frame(self):
        Oad_idx = self._get_Oad_idx()
        Oad_idx = Oad_idx.flatten()
        Oad_cn = self._get_Oad_H_cn(Oad_idx)
        Oad_H_idx = Oad_idx[Oad_cn == 1]
        M5c_OadH_idx = self.M5c_idx[Oad_cn == 1]
        Had_idx = self._get_Had_idx(Oad_H_idx)
        num_surf_spec = len(Oad_H_idx)

        bond_O_H = self._get_Oad_Had_dist(Oad_H_idx, Had_idx)
        angle_M_O_H = self._get_Ti5c_Oad_Had_angle(
            M5c_OadH_idx, Oad_H_idx, Had_idx)

        self.all_Oad_H_idx[self._frame_index, :num_surf_spec] = Oad_H_idx
        self.all_M5c_OadH_idx[self._frame_index, :num_surf_spec] = M5c_OadH_idx
        self.all_Had_idx[self._frame_index, :num_surf_spec] = Had_idx
        self.all_bond_O_H[self._frame_index, :num_surf_spec] = bond_O_H
        self.all_angle_M_O_H[self._frame_index, :num_surf_spec] = angle_M_O_H

    def _conclude(self):
        # save into a compress zip npy
        np.savez_compressed(
            self.fn_all_info,
            all_Oad_H_idx=self.all_Oad_H_idx,
            all_M5c_OadH_idx=self.all_M5c_OadH_idx,
            all_Had_idx=self.all_Had_idx,
            all_bond_O_H=self.all_bond_O_H,
            all_angle_M_O_H=self.all_angle_M_O_H
        )

    def _get_Oad_idx(self, n_ow=1):
        pos = self._ag.positions
        M5c_idx = self.M5c_idx
        Ow_idx = self.Ow_idx
        cellpar = self.cellpar
        M5c_Ow_darray = distance_array(pos[M5c_idx], pos[Ow_idx], box=cellpar)
        sort_idx = np.argsort(M5c_Ow_darray, axis=1)
        Oad_idx = Ow_idx[sort_idx[:, :n_ow]]
        return Oad_idx

    def _get_Oad_H_cn(self, Oad_idx):
        pos = self._ag.positions
        Oad_cn = count_cn(pos[Oad_idx], pos[self.H_idx],
                          1.2, None, cell=self.cellpar)
        return Oad_cn

    def _get_Had_idx(self, Oad_H_idx):
        pos = self._ag.positions
        # Had is H atom in OadH
        OadH_H_darray = distance_array(
            pos[Oad_H_idx], pos[self.H_idx], box=self.cellpar)
        sort_idx = np.argsort(OadH_H_darray, axis=1)
        Had_idx = self.H_idx[sort_idx[:, :1]]
        return Had_idx.flatten()

    def _get_Oad_Had_dist(self, Oad_H_idx, Had_idx):
        pos = self._ag.positions
        dist = calc_bonds(pos[Oad_H_idx], pos[Had_idx], box=self.cellpar)
        return dist

    def _get_Ti5c_Oad_Had_angle(self, M5c_idx, Oad_H_idx, Had_idx):
        pos = self._ag.positions
        angle = calc_angles(pos[M5c_idx], pos[Oad_H_idx],
                            pos[Had_idx], box=self.cellpar)
        return angle
