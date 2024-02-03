# author:  brh
# summary: rutile (110)-water slab and interface methods
# 2021-2022

# Fater classes
from cgi import test
from .slab import Slab
from .interface import Interface

import warnings

# general modules
import numpy as np
from scipy.spatial import distance_matrix

# distance caculation library functions and utilities
from MDAnalysis.lib.distances import (capped_distance,
                                      minimize_vectors)
from ..utils.rutile110 import (get_rotM,
                               get_rotM_edged_rutile110,
                               get_sym_edge,
                               get_watOidx,
                               count_cn,
                               sep_upper_lower,
                               interface_2_slab)
from ..utils.math import (fit_plane_normal,
                          fit_line_vec)


class SlabRutile110(Slab):
    """Slab object for rutile 110 slabs

    Args:
        Slab (Slab): Child class of ASE Atoms class.
    """

    def __init__(self, slab, M="Ti", nrow=2, cutoff=2.8, bridge_along="y"):
        """Init rutile 110 slab

        Args:
            slab (ASE atoms):
                ASE atoms object for a rutile (110) slab model
            M (str, optional):
                The metal element in the model. Defaults to "Ti".
            nrow (int, optional):
                Nuber of Obr rows in the slab model. Defaults to 2.
            cutoff (float, optional):
                Cutoff distances for Ti-O bond. Used for coordination number calculation. Defaults to 2.8.
            bridge_along (str, optional):
                "x" or "y". The direction of Obr rows (i.e. [001] direction). Defaults to "y".
        """
        self.slab = slab
        super().__init__(slab)
        self.cellpar = slab.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.cutoff = cutoff
        self.bridge_along = bridge_along
        # positions and indices
        self.xyz = slab.positions
        self.idx_H = np.where(slab.symbols == "H")[0]
        self.idx_O = np.where(slab.symbols == "O")[0]
        self.idx_M = np.where(slab.symbols == self.M)[0]
        # result dictionary:
        # the keys are the names of the special atoms
        # the values are the corresponding indices
        # the indices array looks like [[<idx_upper>], [idx_lower]]
        self.indices = {
            "idx_M5c": None,
            "idx_Obr": None,
        }
        self.get_surface_indices()

    def get_surf_ti(self):
        cn = count_cn(self.xyz[self.idx_M], self.xyz[self.idx_O],
                      cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_M5c = self.idx_M[cn == 5]
        return idx_M5c

    def get_surf_o(self):
        cn = count_cn(self.xyz[self.idx_O], self.xyz[self.idx_M],
                      cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_Obr2 = self.idx_O[cn == 2]
        return idx_Obr2

    def get_surface_indices(self):
        idx_M5c = self.get_surf_ti()
        idx_Obr2 = self.get_surf_o()
        res = [idx_M5c,
               idx_Obr2]
        for ii, key in enumerate(self.indices):
            self.indices[key] = res[ii]
        return self.indices

    def sep_upper_lower(self):
        for key in self.indices:
            self.indices[key] = sep_upper_lower(self.xyz[:, -1],
                                                 self.indices[key])
        self.ngroup = self.indices['idx_M5c'].shape[-1]//self.nrow
        return self.indices

    def sort_idx(self):
        for key in self.indices:
            tmp = []
            for ii in range(2):
                idx_sorted = sort_by_rows(self.xyz, self.indices[key][ii],
                                          rotM=None, n_row=self.nrow,
                                          bridge_along=self.bridge_along)
                tmp.append(idx_sorted)
            self.indices[key] = np.array(tmp)
        return self.indices


class Rutile110(Interface):
    """Class for rutile (110)-water interface model

    Args:
        Interface (Interface): A child class for ASE Atoms. Representing solid-water interface.
    """

    def __init__(self, atoms, M="Ti", nrow=2, cutoff=2.8, bridge_along="y"):
        """Initialize rutile110

        Args:
            slab (ASE atoms):
                ASE atoms object for a rutile (110) slab model
            M (str, optional):
                The metal element in the model. Defaults to "Ti".
            nrow (int, optional):
                Nuber of Obr rows in the slab model. Defaults to 2.
            cutoff (float, optional):
                Cutoff distances for Ti-O bond. Used for coordination number calculation. Defaults to 2.8.
            bridge_along (str, optional):
                "x" or "y". The direction of Obr rows (i.e. [001] direction). Defaults to "y".
        """
        self.atoms = atoms
        super().__init__(atoms)
        self.cellpar = atoms.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.cutoff = cutoff
        self.bridge_along = bridge_along
        # positions and indices
        self.xyz = atoms.positions
        self.idx_H = np.where(atoms.symbols == "H")[0]
        self.idx_O = np.where(atoms.symbols == "O")[0]
        self.idx_M = np.where(atoms.symbols == self.M)[0]
        self.idx_Ow, _ = self.get_wat()

        # get corresponding slab model
        self.idx_slab, self.slab = self.get_slab_obj()
        self.slab.sep_upper_lower()
        self.slab.sort_idx()

        # convert slab indices to indices edge model
        self.indices = self.get_indices()

    def get_slab_obj(self):
        idx_slab, slab = interface_2_slab(self.atoms, self.M)
        obj = SlabRutile110(slab,
                            M=self.M,
                            nrow=self.nrow,
                            cutoff=self.cutoff,
                            bridge_along=self.bridge_along)
        return idx_slab, obj

    def get_wat(self):
        idx_Ow, idx_H = get_watOidx(self.atoms, M=self.M)
        return idx_Ow, idx_H

    def get_indices(self):
        indices = self.slab.indices.copy()
        for key in indices:
            indices[key] = self.idxslab2idxatoms(indices[key], self.idx_slab)
        return indices

    @staticmethod
    def idxslab2idxatoms(idx_target, idx_slab):
        return idx_slab[idx_target]


class SlabRutile1p11Edge(Slab):
    """Slab object for rutile (110) slab with <1 -1 1> step edge

    Args:
        Slab (Slab): A child class of ASE Atoms. Representing slab model.
    """

    def __init__(self, slab, rotM, M="Ti", nrow=2, cutoff=2.8, bridge_along="y"):
        """intitialize 'Slab' object for rutile (110) slab with <1 -1 1> step edge

        Args:
            slab (ASE atoms):
                ASE atoms object for a rutile (110) slab model
            rotM (numpy.ndarray):
                3x3 numpy array. A rotation matrix for triclicnic simulation box.
                To get positions after rotation, use 'np.matmul(slab.positions, rotM)'.
                The results should be z-Axis of the triclicnic simulation box parallel to [1 1 0].
            M (str, optional):
                The metal element in the model. Defaults to "Ti".
            nrow (int, optional):
                Nuber of Obr rows in the slab model. Defaults to 2.
            cutoff (float, optional):
                Cutoff distances for Ti-O bond. Used for coordination number calculation. Defaults to 2.8.
            bridge_along (str, optional):
                "x" or "y". The direction of Obr rows (i.e. [001] direction). Defaults to "y".
        """
        self.slab = slab
        super().__init__(slab)
        self.rotM = rotM
        self.cellpar = slab.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.cutoff = cutoff
        self.bridge_along = bridge_along
        # positions and indices
        self.xyz = slab.positions
        self.idx_H = np.where(slab.symbols == "H")[0]
        self.idx_O = np.where(slab.symbols == "O")[0]
        self.idx_M = np.where(slab.symbols == self.M)[0]
        # result dictionary:
        # the keys are the names of the special atoms
        # the values are the corresponding indices
        # the indices array looks like [[<idx_upper>], [idx_lower]]
        self.indices = {
            "idx_M5c": None,
            "idx_edge_M5c": None,
            "idx_edge_M4c": None,
            "idx_Obr": None,
            "idx_hObr_mid": None,
            "idx_hObr_upper": None,
            "idx_edge_O2": None
        }
        self.get_surface_indices()

    def get_surf_ti(self):
        cn = count_cn(self.xyz[self.idx_M], self.xyz[self.idx_O],
                      cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_edge4 = self.idx_M[cn == 4]
        idx_edge5, idx_surf5 = self.sep_ticn5(self.slab, self.xyz, self.idx_O,
                                              self.idx_M, cn)
        return idx_edge4, idx_edge5, idx_surf5

    def get_surf_o(self):
        cn = count_cn(self.xyz[self.idx_O], self.xyz[self.idx_M],
                      cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_edge2, idx_hobr, idx_obr = self.sep_ocn2(self.slab, self.xyz,
                                                     self.idx_O, self.idx_M, cn)
        return idx_edge2, idx_hobr, idx_obr

    def get_surface_indices(self):
        idx_edge4, idx_edge5, idx_surf5 = self.get_surf_ti()
        idx_edge2, idx_hobr, idx_obr = self.get_surf_o()
        idx_hobr1, idx_hobr2 = self.sep_hobr(self.slab, idx_hobr, self.M)
        res = [idx_surf5, idx_edge5, idx_edge4,
               idx_obr, idx_hobr1, idx_hobr2, idx_edge2]
        for ii, key in enumerate(self.indices):
            # res[ii] = sep_upper_lower(self.xyz[:,-1], res[ii])
            self.indices[key] = res[ii]
        return self.indices

    def sep_upper_lower(self):
        for key in self.indices:
            self.indices[key] = sep_upper_lower(self.xyz[:, -1],
                                                 self.indices[key])
        self.ngroup = self.indices['idx_M5c'].shape[-1]//self.nrow
        return self.indices

    def sort_idx(self):
        for key in self.indices:
            tmp = []
            for ii in range(2):
                idx_sorted = sort_by_rows(self.xyz, self.indices[key][ii],
                                          rotM=self.rotM, n_row=self.nrow,
                                          bridge_along=self.bridge_along)
                tmp.append(idx_sorted)
            self.indices[key] = np.array(tmp)
        return self.indices

    def sep_ticn5(self, slab, xyz, idx_O, idx_M, cn):
        pairs, _ = capped_distance(xyz[idx_M], xyz[idx_O], self.cutoff, None,
                                   box=slab.cell.cellpar())
        idx_cn5 = np.where(cn == 5)[0]
        sel = np.zeros(pairs.shape[0])
        for cn5 in idx_cn5:
            sel += (pairs[:, 0] == cn5).astype(int)
        cn5_pairs = pairs[:][sel.astype(bool)]
        cno = count_cn(xyz[idx_O[cn5_pairs[:, -1]]], xyz[idx_M], cutoff_hi=self.cutoff,
                       cutoff_lo=None, cell=slab.cell.cellpar())
        cno_sum = cno.reshape(-1, 5).sum(axis=1)
        idx_edge5 = idx_M[idx_cn5[cno_sum == 12]]
        idx_surf5 = idx_M[idx_cn5[cno_sum == 15]]
        return idx_edge5, idx_surf5

    def sep_ocn2(self, slab, xyz, idx_O, idx_M, cn):
        pairs, _ = capped_distance(xyz[idx_O], xyz[idx_M], self.cutoff, None,
                                   box=slab.cell.cellpar())
        idx_cn2 = np.where(cn == 2)[0]
        sel = np.zeros(pairs.shape[0])
        for cn2 in idx_cn2:
            sel += (pairs[:, 0] == cn2).astype(int)
        cn2_pairs = pairs[:][sel.astype(bool)]
        cnti = count_cn(xyz[idx_M[cn2_pairs[:, -1]]], xyz[idx_O], cutoff_hi=self.cutoff,
                        cutoff_lo=None, cell=slab.cell.cellpar())
        cnti_sum = cnti.reshape(-1, 2).sum(axis=1)
        idx_edge2 = idx_O[idx_cn2[cnti_sum == 9]]
        idx_hobr = idx_O[idx_cn2[cnti_sum == 11]]
        idx_obr = idx_O[idx_cn2[cnti_sum == 12]]
        return idx_edge2, idx_hobr, idx_obr

    @staticmethod
    def sep_hobr(atoms, idx_hobr, M="Ti"):
        trig = get_triangle(atoms, idx_hobr, M)
        vec = trig_vec(atoms, trig)
        hobr1 = idx_hobr[np.abs(np.dot(vec, [0, 0, 1])) < 2]
        hobr2 = idx_hobr[np.abs(np.dot(vec, [0, 0, 1])) > 2]
        return hobr1, hobr2


class Rutile1p11Edge(Interface):
    """Interface object for rutile (110) with <1 -1 1> edge-water interface model.

    Args:
        Interface (ASE Atoms): A child class of ASE atoms
    """
    def __init__(self, atoms, vecy=None, vecz=None, M="Ti", nrow=2, cutoff=2.8, bridge_along="y"):
        """Initialize `Interface` object for rutile (110) with <1 -1 1> edge-water interface

        Args:
            slab (ASE atoms):
                ASE atoms object for a rutile (110) slab model
            vecy (numpy.ndarray, optional):
                (3, )-shaped numpy array. Could be any vector parallel to Obr rows ([001] direction).
                defualts to None.
            vecz (numpy.ndarray, optional):
                (3, )-shaped numpy array. Could be any vector parallel to [110] direction.
                defualts to None.
            M (str, optional):
                The metal element in the model. Defaults to "Ti".
            nrow (int, optional):
                Nuber of Obr rows in the slab model. Defaults to 2.
            cutoff (float, optional):
                Cutoff distances for Ti-O bond. Used for coordination number calculation. Defaults to 2.8.
            bridge_along (str, optional):
                "x" or "y". The direction of Obr rows (i.e. [001] direction). Defaults to "y".
        """
        self.atoms = atoms
        super().__init__(atoms)
        self.cellpar = atoms.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.cutoff = cutoff
        self.bridge_along = bridge_along
        # tanslate the cell first
        self.atoms = get_sym_edge(self.atoms)
        # positions and indices
        self.xyz = atoms.positions
        self.idx_H = np.where(atoms.symbols == "H")[0]
        self.idx_O = np.where(atoms.symbols == "O")[0]
        self.idx_M = np.where(atoms.symbols == self.M)[0]
        self.idx_Ow, _ = self.get_wat()

        # get_rotation matrix
        if (vecy is not None) and (vecz is not None):
            self.rotM = get_rotM(vecy, vecz)
        else:
            tmp = get_rotM_edged_rutile110(atoms, bridge_along=bridge_along)
            if bridge_along == "y":
                vecy, vecz = tmp[1], tmp[2]
            elif bridge_along == "x":
                vecy, vecz = tmp[0], tmp[2]
            else:
                raise ValueError(f"The value for 'bridge_along' could only be 'x' or 'y'. However, you provided '{bridge_along}'.")
            self.rotM = get_rotM(vecy, vecz)

        # get corresponding slab model
        self.idx_slab, self.slab = self.get_slab_obj()
        self.slab.sep_upper_lower()
        self.slab.sort_idx()

        # convert slab indices to indices edge model
        self.indices = self.get_indices()

    def get_slab_obj(self):
        idx_slab, slab = interface_2_slab(self.atoms, self.M)
        obj = SlabRutile1p11Edge(slab,
                                 rotM=self.rotM,
                                 M=self.M,
                                 nrow=self.nrow,
                                 cutoff=self.cutoff,
                                 bridge_along=self.bridge_along)
        return idx_slab, obj

    def get_wat(self):
        idx_Ow, idx_H = get_watOidx(self.atoms, M=self.M)
        return idx_Ow, idx_H

    def get_indices(self):
        """
        Returns a dictionary of atom indices in the slab.

        This method creates a copy of the `indices` dictionary from the `slab` attribute.
        It then updates each value in the dictionary by converting the slab index to the corresponding atom index.

        Returns:
            dict: A dictionary where the keys are atom types and the values are lists of atom indices.

        Example:
            Assuming `rutile` is an instance of the `Rutile110` class:

            >>> indices = rutile.get_indices()
            >>> print(indices)
            {'O': [1, 2, 3, 4], 'Ti': [5, 6, 7, 8]}

            This will print a dictionary where the keys are atom types ('O' and 'Ti' in this case)
            and the values are lists of atom indices.
        """
        indices = self.slab.indices.copy()
        for key in indices:
            indices[key] = self.idxslab2idxatoms(indices[key], self.idx_slab)
        return indices

    def refine_rotM(self):
        _vecy = self._refine_vecy()
        _vecz = self._refine_vecz()
        testy = np.isnan(_vecy).sum()
        testz = np.isnan(_vecz).sum()
        if (testy+testz) > 0:
            warnings.warn("warning! couldn't refine rotM for some reason. Return the original rotM instead")
            return self.rotM
        else:
            return get_rotM(_vecy, _vecz)

    def _refine_vecz(self):
        ind = self.get_indices()
        xyz = self.positions
        nTi_per_row = ind['idx_M5c'].shape[-1]
        idx_upperM5c, idx_lowerM5c = ind['idx_M5c'].reshape(
            2, self.nrow*nTi_per_row)
        nObr_per_row = ind['idx_Obr'].shape[-1]
        idx_upperObr, idx_lowerObr = ind['idx_Obr'].reshape(
            2, self.nrow*nObr_per_row)
        plane_list = (idx_upperM5c, idx_lowerM5c, idx_upperObr, idx_lowerObr)
        res = np.empty((len(plane_list), 3), dtype=float)
        for ii, ind in enumerate(plane_list):
            res[ii] = fit_plane_normal(xyz[ind])
        res = res.mean(axis=0)
        _vecz = res/np.linalg.norm(res)
        return _vecz

    def _refine_vecy(self):
        ind = self.get_indices()
        xyz = self.positions
        nTi_per_row = ind['idx_M5c'].shape[-1]
        idx_M5c = ind['idx_M5c'].reshape(-1, nTi_per_row)
        nObr_per_row = ind['idx_Obr'].shape[-1]
        idx_Obr = ind['idx_Obr'].reshape(-1, nObr_per_row)
        line_list = np.concatenate([idx_M5c, idx_Obr], axis=0)
        res = np.empty((line_list.shape[0], 3), dtype=float)
        for ii, ind in enumerate(line_list):
            tmp = fit_line_vec(xyz[ind])
            if tmp[0] < 0:
                tmp = -1*tmp
            if tmp[0] < 0.98:
                res[ii] = np.ones(3) * np.nan
            else:
                res[ii] = tmp
        res = np.nanmean(res, axis=0)
        _vecy = res/np.linalg.norm(res)
        return _vecy

    @staticmethod
    def idxslab2idxatoms(idx_target, idx_slab):
        return idx_slab[idx_target]

def sort_by_rows(xyz, idx, rotM=None, n_row=2, bridge_along="y"):
    """ Sorting surface atoms in the (110) row, particularly the M5c's and Obr's row-wise.

    Algorithm description:
    (NB: the rotation matrix assumes: bridge_along="y". See this convention in the constructor for 'Rutile1p11Edge')
    1. save the (xx, yy) coordinates with and without rotation (rotation matrix is not required for for a flat model)
    2. the 'make_groups' method will group rows together using the (rotated) x coordinates:
    ^ y
    |
    |---------     On the left we have and simple demonstration.
    | *    + |     Due to the nature of a triclinic box,
    |  *    +|     single rows, denoted by symbols "+" and "*",
    | + * ...|     might be wrapped to the other end of the box.
    |  + *   |     The 'make_groups' function essentially distinguishes
    |   + *  |     "+" and "*". Specifically, the method does something like:
    |    + * |           make_groups(array("+", "*")) = array(array("+"), array("*"))
    |------------> x
    3. the collection of three methods, namely, "group_each_row", "sort_each_row", and "sort_grouped_rows",
    are used to sort each row. For rows without "discontinuity" ("*"), method 'sort_each_row' will suffice.
    However, complexity arise at discontinuity. For example, if we use "o" to denote a discontinuity, we observe
    ^ y               that after apply the rotation matridx, the two "o" will have the nearly identical 'yy_rot' value.
    |                 Hence, in may cases, sorting of this discontinuity is not straitforward. Fortuanetely, this problem is solved
    |---------        if we first treat different segments of the "+" row separately, and for the discontinuity, we use the original
    | *    + |        coordinates to sort the discontinuity (o). A implementation of this algorithm is given in 'sort_grouped_rows'
    |  *    o|
    | o * ...|
    |  + *   |
    ..............

    """

    # Firsrt: rotate xyz and cell param
    if bridge_along == "y":
        xx, yy = xyz[:, 0].copy(), xyz[:, 1].copy()
    elif bridge_along == "x":
        yy, xx = xyz[:, 0].copy(), xyz[:, 1].copy()
    else:
        raise ValueError(f"bridge_along should be either 'x' or 'y'. However, you have {bridge_along}")

    if rotM is not None:
        xyz_rot = np.matmul(xyz, rotM)
        xx_rot, yy_rot = xyz_rot[:, 0].copy(), xyz_rot[:, 1].copy()
    else:
        xx_rot, yy_rot = xx, yy

    # Second: group the rows
    def make_groups(idx, xx):
        dm = distance_matrix(xx[idx].reshape(-1, 1), xx[idx].reshape(-1, 1))
        groups = np.unique(dm <= 2, axis=0)

        if groups.shape[0] == n_row:
            pass
        else:
            sel = (groups.sum(axis=1) < dm.shape[0] // n_row)
            merge = np.array(groups[sel].sum(axis=0)).astype(bool)[:, np.newaxis].T
            groups = np.concatenate([groups[~sel], merge], axis=0)
        rows = [idx[groups[ii, :]] for ii in range(n_row)]
        return rows


    # Third: algorithms to sort each row
    def group_each_row(id, coords, ):
        dm = distance_matrix(coords[id].reshape(-1, 1), coords[id].reshape(-1, 1))
        dm_coarse = np.round(dm)
        groups = np.unique(dm_coarse <= 2, axis=0)

        return groups

    # Thrid sort along x axis
    def sort_each_row(row, coords):
        return row[np.argsort(coords[row])]

    def sort_grouped_rows(id, groups, coords, coords_rot):
        if len(groups) == 1:
            return sort_each_row(id, coords_rot)

        else:
            rows = []
            first_elements_coords = []

            for ig, g in enumerate(groups):
                row_ig = sort_each_row(id[g], coords_rot)
                rows.append(row_ig)
                first_elements_coords.append(coords[row_ig[0]])

            argsort = np.argsort(first_elements_coords)
            rows = np.array(rows, dtype=object)[argsort]
            rows = np.concatenate(rows).astype(int)
            return rows

    idx_sorted_perpendicular_bridge = make_groups(idx, xx_rot)
    res = np.zeros_like(idx_sorted_perpendicular_bridge)

    for irow in range(n_row):
        id = idx_sorted_perpendicular_bridge[irow]
        groups = group_each_row(id, xx_rot)
        res[irow] = sort_grouped_rows(id, groups, yy, yy_rot)

    return res

def get_triangle(atoms, idx_obr, M="Ti", cutoff=2.8):
    """What the heck is this?
    """
    idx_M = np.where(atoms.symbols == M)[0]
    xyz = atoms.positions
    pairs, _ = capped_distance(xyz[idx_obr], xyz[idx_M], max_cutoff=cutoff,
                               min_cutoff=None, box=atoms.cell.cellpar())
    res = pairs[:, -1].reshape(-1, 2)
    res = np.concatenate([idx_obr[:, np.newaxis], idx_M[res]], axis=1)
    return res


def trig_vec(atoms, trig):
    """What the heck is this?
    """
    xyz = atoms.positions
    idx_O = trig[:, 0]
    idx_M1 = trig[:, 2]
    idx_M2 = trig[:, 1]
    v1 = minimize_vectors(xyz[idx_M1] - xyz[idx_O], box=atoms.cell.cellpar())
    v2 = minimize_vectors(xyz[idx_M2] - xyz[idx_O], box=atoms.cell.cellpar())
    v = np.round(v1+v2, 1)
    return v
