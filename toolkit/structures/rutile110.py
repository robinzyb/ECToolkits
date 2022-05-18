# author:  brh
# summary: rutile (110)-water slab and interface methods
# 2021-2022

# Fater classes 
from .slab import Slab
from .interface import Interface

# general modules
import numpy as np
from scipy.spatial import distance_matrix

# distance caculation library functions and utilities
from MDAnalysis.lib.distances import (capped_distance,
                                      minimize_vectors)
from ..utils.rutile110 import (get_rotM,
                               get_sym_edge,
                               get_watOidx,
                               count_cn,
                               sep_upper_lower,
                               interface_2_slab)

class SlabRutile110(Slab):

    def __init__(self, slab, M="Ti", nrow=2, cutoff=2.8, bridge_along="y"):
        self.slab = slab
        super().__init__(slab)
        self.cellpar = slab.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.cutoff = cutoff 
        self.bridge_along = bridge_along
        # positions and indicies
        self.xyz    = slab.positions
        self.idx_H  = np.where(slab.symbols=="H")[0]
        self.idx_O  = np.where(slab.symbols=="O")[0]
        self.idx_M = np.where(slab.symbols==self.M)[0]
        # result dictionary: 
        # the keys are the names of the special atoms
        # the values are the corresponding indicies
        # the indicies array looks like [[<idx_upper>], [idx_lower]]
        self.indicies = {
            "idx_M5c": None,
            "idx_Obr": None,
            }
        self.get_surface_indicies()
            
    def get_surf_ti(self):
        cn = count_cn(self.xyz[self.idx_M], self.xyz[self.idx_O], 
            cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_M5c = self.idx_M[cn==5]
        return idx_M5c

    def get_surf_o(self):
        cn = count_cn(self.xyz[self.idx_O], self.xyz[self.idx_M], 
            cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_Obr2 = self.idx_O[cn==2] 
        return idx_Obr2
    
    def get_surface_indicies(self): 
        idx_M5c = self.get_surf_ti()
        idx_Obr2 = self.get_surf_o()
        res = [idx_M5c, 
               idx_Obr2]
        for ii, key in enumerate(self.indicies):
            self.indicies[key] = res[ii]
        return self.indicies

    def sep_upper_lower(self):
        for key in self.indicies:
            self.indicies[key] = sep_upper_lower(self.xyz[:,-1],
                    self.indicies[key])
        self.ngroup = self.indicies['idx_M5c'].shape[-1]//self.nrow
        return self.indicies

    def sort_idx(self):
        for key in self.indicies:
            tmp = []
            for ii in range(2):
                idx_sorted = sort_by_rows(self.slab, self.indicies[key][ii],
                        rotM=None, n_row=self.nrow,
                        bridge_along=self.bridge_along)
                tmp.append(idx_sorted)
            self.indicies[key] = np.array(tmp)
        return self.indicies

class Rutile110(Interface):

    def __init__(self, atoms, M="Ti", nrow=2, bridge_along="y"):
        self.atoms = atoms
        super().__init__(atoms)
        self.cellpar = atoms.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.bridge_along = bridge_along
        # positions and indicies
        self.xyz    = atoms.positions
        self.idx_H  = np.where(atoms.symbols=="H")[0]
        self.idx_O  = np.where(atoms.symbols=="O")[0]
        self.idx_M = np.where(atoms.symbols==self.M)[0]
        self.idx_Ow, _ = self.get_wat()

        # get corresponding slab model
        self.idx_slab, self.slab = self.get_slab_obj()
        self.slab.sep_upper_lower() 
        self.slab.sort_idx()

        # convert slab indicies to indicies edge model
        self.indicies = self.get_indicies()
    
    def get_slab_obj(self):
        idx_slab, slab = interface_2_slab(self.atoms, self.M)
        obj = SlabRutile110(slab, 
                M=self.M, 
                nrow=self.nrow,
                bridge_along=self.bridge_along)
        return idx_slab, obj

    def get_wat(self):
        idx_Ow, idx_H = get_watOidx(self.atoms, M=self.M)
        return idx_Ow, idx_H

    def get_indicies(self):
        indicies = self.slab.indicies.copy()
        for key in indicies:
            indicies[key] = self.idxslab2idxatoms(indicies[key], self.idx_slab)
        return indicies

    @staticmethod
    def idxslab2idxatoms(idx_target, idx_slab):
        return idx_slab[idx_target]

class SlabRutile1p11Edge(Slab):

    def __init__(self, slab, rotM, M="Ti", nrow=2, cutoff=2.8, bridge_along="y"):
        self.slab = slab
        super().__init__(slab)
        self.rotM = rotM
        self.cellpar = slab.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.cutoff=cutoff
        self.bridge_along = bridge_along
        # positions and indicies
        self.xyz    = slab.positions
        self.idx_H  = np.where(slab.symbols=="H")[0]
        self.idx_O  = np.where(slab.symbols=="O")[0]
        self.idx_M = np.where(slab.symbols==self.M)[0]
        # result dictionary: 
        # the keys are the names of the special atoms
        # the values are the corresponding indicies
        # the indicies array looks like [[<idx_upper>], [idx_lower]]
        self.indicies = {
            "idx_M5c": None,
            "idx_edge_M5c": None,
            "idx_edge_M4c": None,
            "idx_Obr": None,
            "idx_hObr_mid": None,
            "idx_hObr_upper": None,
            "idx_edge_O2": None
            }
        self.get_surface_indicies()
            
    def get_surf_ti(self):
        cn = count_cn(self.xyz[self.idx_M], self.xyz[self.idx_O], 
            cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_edge4 = self.idx_M[cn==4]
        idx_edge5, idx_surf5 = self.sep_ticn5(self.slab, self.xyz, self.idx_O, 
            self.idx_M, cn)
        return idx_edge4, idx_edge5, idx_surf5

    def get_surf_o(self):
        cn = count_cn(self.xyz[self.idx_O], self.xyz[self.idx_M], 
            cutoff_hi=self.cutoff, cutoff_lo=None, cell=self.cellpar)
        idx_edge2, idx_hobr, idx_obr = self.sep_ocn2(self.slab, self.xyz, 
            self.idx_O, self.idx_M, cn)
        return idx_edge2, idx_hobr, idx_obr
    
    def get_surface_indicies(self): 
        idx_edge4, idx_edge5, idx_surf5 = self.get_surf_ti()
        idx_edge2, idx_hobr, idx_obr = self.get_surf_o() 
        idx_hobr1, idx_hobr2 = self.sep_hobr(self.slab, idx_hobr, self.M)
        res = [idx_surf5, idx_edge5, idx_edge4, 
               idx_obr, idx_hobr1, idx_hobr2, idx_edge2]
        for ii, key in enumerate(self.indicies):
            #res[ii] = sep_upper_lower(self.xyz[:,-1], res[ii])
            self.indicies[key] = res[ii]
        return self.indicies

    def sep_upper_lower(self):
        for key in self.indicies:
            self.indicies[key] = sep_upper_lower(self.xyz[:,-1],
                    self.indicies[key])
        self.ngroup = self.indicies['idx_M5c'].shape[-1]//self.nrow
        return self.indicies
    
    def sort_idx(self):
        for key in self.indicies:
            tmp = []
            for ii in range(2):
                idx_sorted = sort_by_rows(self.slab, self.indicies[key][ii], 
                                          rotM=self.rotM, n_row=self.nrow,
                                          bridge_along=self.bridge_along)
                tmp.append(idx_sorted)
            self.indicies[key] = np.array(tmp)
        return self.indicies

    def sep_ticn5(self, slab, xyz, idx_O, idx_M, cn):
        pairs, _ = capped_distance(xyz[idx_M], xyz[idx_O], self.cutoff, None,
                                           box=slab.cell.cellpar())
        idx_cn5 = np.where(cn==5)[0]
        sel = np.zeros(pairs.shape[0])
        for cn5 in idx_cn5:
            sel += (pairs[:, 0]==cn5).astype(int)
        cn5_pairs = pairs[:][sel.astype(bool)]
        cno = count_cn(xyz[idx_O[cn5_pairs[:, -1]]], xyz[idx_M], cutoff_hi=self.cutoff,
                       cutoff_lo=None, cell=slab.cell.cellpar())
        cno_sum = cno.reshape(-1, 5).sum(axis=1)
        idx_edge5 = idx_M[idx_cn5[cno_sum==12]]
        idx_surf5 = idx_M[idx_cn5[cno_sum==15]]
        return idx_edge5, idx_surf5

    def sep_ocn2(self, slab, xyz, idx_O, idx_M, cn):
        pairs, _ = capped_distance(xyz[idx_O], xyz[idx_M], self.cutoff, None,
                                           box=slab.cell.cellpar())
        idx_cn2 = np.where(cn==2)[0]
        sel = np.zeros(pairs.shape[0])
        for cn2 in idx_cn2:
            sel += (pairs[:, 0]==cn2).astype(int)
        cn2_pairs = pairs[:][sel.astype(bool)]
        cnti = count_cn(xyz[idx_M[cn2_pairs[:, -1]]], xyz[idx_O], cutoff_hi=self.cutoff,
                       cutoff_lo=None, cell=slab.cell.cellpar())
        cnti_sum = cnti.reshape(-1, 2).sum(axis=1)
        idx_edge2 = idx_O[idx_cn2[cnti_sum==9]]
        idx_hobr = idx_O[idx_cn2[cnti_sum==11]]
        idx_obr = idx_O[idx_cn2[cnti_sum==12]]
        return idx_edge2, idx_hobr, idx_obr
        
    @staticmethod
    def sep_hobr(atoms, idx_hobr, M="Ti"):
        trig = get_triangle(atoms, idx_hobr, M)
        vec = trig_vec(atoms, trig)
        hobr1 = idx_hobr[np.abs(np.dot(vec, [0, 0, 1])) < 2]
        hobr2 = idx_hobr[np.abs(np.dot(vec, [0, 0, 1])) > 2]
        return hobr1, hobr2 


class Rutile1p11Edge(Interface):

    def __init__(self, atoms, vecy, vecz, M="Ti", nrow=2, bridge_along="y"):
        self.atoms = atoms
        super().__init__(atoms)
        self.cellpar = atoms.cell.cellpar()
        self.M = M
        self.nrow = nrow
        self.bridge_along = bridge_along
        # tanslate the cell first
        self.atoms = get_sym_edge(self.atoms)
        # positions and indicies
        self.xyz    = atoms.positions
        self.idx_H  = np.where(atoms.symbols=="H")[0]
        self.idx_O  = np.where(atoms.symbols=="O")[0]
        self.idx_M = np.where(atoms.symbols==self.M)[0]
        self.idx_Ow, _ = self.get_wat()

        # get_rotation matrix
        self.rotM = get_rotM(vecy, vecz) 
        # get corresponding slab model
        self.idx_slab, self.slab = self.get_slab_obj()
        self.slab.sep_upper_lower() 
        self.slab.sort_idx()

        # convert slab indicies to indicies edge model
        self.indicies = self.get_indicies()
    
    def get_slab_obj(self):
        idx_slab, slab = interface_2_slab(self.atoms, self.M)
        obj = SlabRutile1p11Edge(slab, 
                rotM=self.rotM,
                M=self.M, 
                nrow=self.nrow,
                bridge_along=self.bridge_along)
        return idx_slab, obj

    def get_wat(self):
        idx_Ow, idx_H = get_watOidx(self.atoms, M=self.M)
        return idx_Ow, idx_H

    def get_indicies(self):
        indicies = self.slab.indicies.copy()
        for key in indicies:
            indicies[key] = self.idxslab2idxatoms(indicies[key], self.idx_slab)
        return indicies

    @staticmethod
    def idxslab2idxatoms(idx_target, idx_slab):
        return idx_slab[idx_target]

# non-universal utilities
def sort_by_rows(atoms, idx, rotM=None, n_row=2, bridge_along="y"):
    xyz = atoms.positions
    n_group = idx.shape[0]//n_row
    # FIRST: rotate xyz and cell param
    if rotM is not None:
        xyz = np.matmul(atoms.positions, rotM)
    else:
        xyz = atoms.positions
    # THEN: group ti5c by X-axis (AFTER ROTATE)
    if bridge_along=="x":
        yy, xx = np.round(xyz[:, 0]), np.round(xyz[:, 1])
    elif bridge_along=="y":
        xx, yy = np.round(xyz[:, 0]), np.round(xyz[:, 1])
    dm = distance_matrix(xx[idx].reshape(-1, 1), xx[idx].reshape(-1, 1))
    groups = np.unique(dm<=2, axis=0)

    if groups.shape[0] == n_row:
        pass
    else:
        sel = (groups.sum(axis=1)<n_group)
        merge = np.array(groups[sel].sum(axis=0)).astype(bool)[:, np.newaxis].T
        groups = np.concatenate([groups[~sel], merge], axis=0)
    # LAST: sort ti5c according to Y-axis
    def sort_row(row, yy=yy):
        return row[np.argsort(yy[row])]
    rows = [idx[groups[ii, :]] for ii in range(n_row)]
    cols = np.array(list(map(sort_row, rows)))
    res = cols[np.argsort(xx[cols[:, 0]])]
    return res

def get_triangle(atoms, idx_obr, M="Ti", cutoff=2.8):
    idx_M = np.where(atoms.symbols==M)[0]
    xyz = atoms.positions
    pairs, _ = capped_distance(xyz[idx_obr], xyz[idx_M], max_cutoff=cutoff,
                                       min_cutoff=None, box=atoms.cell.cellpar())
    res = pairs[:, -1].reshape(-1, 2)
    res = np.concatenate([idx_obr[:, np.newaxis], idx_M[res]], axis=1)
    return res

def trig_vec(atoms, trig):
    xyz = atoms.positions
    idx_O = trig[:, 0]
    idx_M1 = trig[:, 2]
    idx_M2 = trig[:, 1]
    v1 = minimize_vectors(xyz[idx_M1] - xyz[idx_O], box=atoms.cell.cellpar())
    v2 = minimize_vectors(xyz[idx_M2] - xyz[idx_O], box=atoms.cell.cellpar())
    v = np.round(v1+v2, 1)
    return v
