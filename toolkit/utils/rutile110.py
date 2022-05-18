# author:  brh
# summary: rutile (110)-water interface toolkit
# summary: this contains the general methods for rutile-water(110)
# 2021-2022

# general modules
import os
import glob
import numpy as np
from ase.geometry import cellpar_to_cell
# distance caculation library functions
from MDAnalysis.lib.distances import *


def count_cn(atoms1, atoms2, cutoff_hi, cutoff_lo=None, cell=None, **kwargs):
    """count the coordination number(CN) for atoms1 (center atoms), where atoms2 are coordinate atom. 
    This function will calculate CN within range cutoff_lo < d < cutoff_lo, where d is the distance 
    between atoms1 and atoms2. Minimum image convention is applied if cell is not `None`

    Args:
        atoms1 (numpy.ndarray): 
            Array with shape (N, 3), where N is the number of center atoms. 'atoms1' are the 
            position of center atoms. 
        atoms2 (numpy.ndarray): 
            Array with shape (M, 3), where M is the number of coordination atoms. 'atoms2' are 
            the positions of coordination atoms.
        cutoff_hi (float): 
            Max cutoff for calculating coordination number. 
        cutoff_lo (float or None, optional): 
            Min cutoff for calculating coordination number. This function will calculate CN within 
            range cutoff_lo < d < cutoff_lo, where d is the distance between atoms1 and atoms2. 
            Defaults to None.
        cell (numpy.ndarray, optional): 
            Array with shape (6,), Array([a, b, c, alpha, beta, gamma]). Simulation cell parameters. 
            If it's not None, the CN calculation will use minimum image convention. Defaults to `None`.

    Returns:
        results: 
            Array with shape (N,), CN of each atoms atoms1
    """
    cell = np.array(cell).astype(np.float32)

    pairs, _ = capped_distance(reference=atoms1,
                               configuration=atoms2,
                               max_cutoff=cutoff_hi,
                               min_cutoff=cutoff_lo,
                               box=cell)
    _minlength = atoms1.shape[0]
    results = np.bincount(pairs[:, 0], minlength=_minlength)
    return results

def cellpar2volume(cellpar):
    cell = cellpar_to_cell(cellpar)
    return np.abs(np.dot(np.cross(cell[0], cell[1]), cell[2]))


def get_watOidx(atoms, M="Ti"):
    """gets all the water oxygen indicies in rutile (110)-water interface

    Args:
        atoms (ase.Atoms): 
            ASE atoms. One snapshot of rutile (110)-water structure.
        M (str, optional): 
            The metal atom in rutile structrue. Defaults to "Ti".

    Returns:
        watOidx (numpy.ndarray): 0-based indicies for water oxygen atoms.
        watHidx (numpy.ndarray): 0-based indicies for water hydrogen atoms. (all the hydrogens)
    """
    xyz = atoms.positions
    cell = atoms.cell.cellpar()
    idx_Ti = np.where(atoms.symbols==M)[0]
    idx_O = np.where(atoms.symbols=='O')[0]
    idx_H = np.where(atoms.symbols=='H')[0]

    cn_H = count_cn(xyz[idx_O, :], xyz[idx_H, :], 1.2, None, cell)
    cn_Ti = count_cn(xyz[idx_O, :], xyz[idx_Ti, :], 2.8, None, cell)

    watOidx = idx_O[(cn_H >= 0) * (cn_Ti <= 1)]
    watHidx = idx_H
    return watOidx, watHidx

def interface_2_slab(atoms, M="Ti"):
    """transform rutile (110)-water interface model to only the slab, i.e., deleting all the 
    water molecules.

    Args:
        atoms (ase.Atoms): 
            ASE atoms. One snapshot of rutile (110)-water structure.
        M (str, optional): 
            The metal atom in rutile structrue. Defaults to "Ti".

    Returns:
        idx_slab(numpy.ndarray): 
            The indicies for the slab model.
        atoms_slab(ase.atoms): 
            Slab model atoms object.
    """
    indicies = np.arange(atoms.get_global_number_of_atoms())
    idx_ow, idx_hw = get_watOidx(atoms, M="Ti")
    idx_wat = np.append(idx_ow, idx_hw)
    idx_slab = np.setdiff1d(indicies, idx_wat)
    atoms_slab = atoms[idx_slab]
    return idx_slab, atoms_slab

def get_rotM(vecy, vecz):
    """get the rotation matrix for rotating a rutile model. Specifically, rotating a step 
    edge model s.t x is parrallel to [001], y is parrallel to [1-10], and z is parallel 
    to \hkl<110>.

    Args:
        vecy (numpy.ndarray): 
            Array with shape (3, ). This direction parallels to Obr/Ti5c direction of the 
            original simulation cell.
        vecz (numpy.ndarray): 
            Array with shape (3, ). This direction parallels to the [110] of the original 
            simulation cell.
    """
    def e_vec(vec):
        return vec/np.linalg.norm(vec)
    vecx = np.cross(vecz, vecy)
    vecx, vecy, vecz = list(map(e_vec, [vecx, vecy, vecz]))
    M = np.array([vecx, vecy, vecz])
    return np.linalg.pinv(M)

def sep_upper_lower(z, indicies):
    """given indicies, seperate them to upper and lower. More specifically, from 
    [<indicies>] to [[<idx_upper>], [<idx_lower>]]

    Args:
        z (numpy.ndarray): 
            z coordinates
        indicies (numpy.ndarray): 
            0-based indicies

    Returns:
        numpy.ndarray: [[<idx_upper>], [<idx_lower>]] 
    """
    z = z[indicies]
    zmean = z.mean()
    i1 = indicies[z>zmean]
    i2 = indicies[z<zmean]
    return np.array([i1, i2])

def pair_M5c_n_obr(atoms, idx_cn5, idx_obrs, M="Ti"):
    """This methods will find the Ti5c's neighboring Obr atoms.

    Args:
        atoms (ASE.Atoms): 
            ase atoms of your interface model.
        idx_cn5 (np.ndarray): 
            1-d integer array, which contains indicies for Ti5c atoms.
        idx_obrs (np.ndarray): 
            1-d integer array, which contains indicies for all the Obr atoms.
        M (str, optional): 
            Metal elements in the rutile structure. Defaults to "Ti".

    Returns:
        tuple: (<idx_cn5>, <res_obr>). 
            Paired Ti5c and Obr indicies. Check if they are really paired before you use.
    """
    # obtain the distance matrix between Ti5c and Obr
    xyz1 = atoms.positions[idx_cn5]
    xyz2 = atoms.positions[idx_obrs]
    dm = distance_array(xyz1, xyz2, box=atoms.cell.cellpar())
    
    # Find the two nearest Obr's to cn5s, and
    # they are exactly the left and right obr
    sort = np.argsort(dm, axis=1)[:, :2]
    res_idx = idx_obrs[sort]
    
    # filp the array s.t.
    # the first column  -> obr POSY
    # the second column -> obr NEGY
    v = atoms.positions[res_idx[:, 0]] - atoms.positions[res_idx[:, 1]]
    mic_v = minimize_vectors(v, box=atoms.cell.cellpar())      # apply minimum image convention
    sel = mic_v[:, 1]<0                           # selective flipping
    res_idx[sel] =  np.flip(res_idx[sel], axis=1)
    return idx_cn5, res_idx



#def group_by_row(atoms, idx, ngroup=2):
#    """Group the atoms by row. To be more specific, rutile (110) surface has rows of Ti5c and Obrs. This method group surface atoms. 
#
#    Args:
#        atoms (ase.Atoms): _description_
#        idx (numpy.ndarray): _description_
#        ngroup (int, optional): . Defaults to 2.
#
#    Raises:
#        ValueError: This method will group the atoms by rows. Throw this error if calculated group number is not consistent with 'ngroup'.
#
#    Returns:
#        numpy.ndarray: [[<row_1>], [<row_2>], ..., [<row_ngroup>]]
#    """
#    xyz = atoms.positions
#    # group the atoms according to their distances
#    dm = np.round(distance_array(xyz[idx], xyz[idx], box=atoms.cell.cellpar()))
#    groups = np.unique(dm < dm.min()+2, axis=0)
#
#    if groups.shape[0] != ngroup:
#        raise ValueError("grouping result does not match 'ngroup'. First check if the row number match 'ngroup'. Consider move your structure")
#
#    rows = idx[groups[0]], idx[groups[1]]
#    return rows
#
#def sort_by_row(atoms, idx, ngroup=2):
#    """Group the atoms by row. To be more specific, rutile (110) surface has rows of Ti5c and Obrs. This method group surface atoms. 
#
#    Args:
#        atoms (ase.Atoms): _description_
#        idx (numpy.ndarray): _description_
#        rotM (numpy.ndarray, optional): The roational matrix. Defaults to None.
#        ngroup (int, optional): . Defaults to 2.
#
#    Raises:
#        ValueError: This method will group the atoms by rows. Throw this error if calculated group number is not consistent with 'ngroup'.
#
#    Returns:
#        _type_: _description_
#    """
#    xyz = atoms.positions
#    # FIRST: rotate xyz
#    if rotM is not None:
#        xyz = np.matmul(atoms.positions, rotM)
#    # THEN: group the atoms having identical x value
#    #xx, yy = np.round(xyz[:, 0]), np.round(xyz[:, 1])
#    dm = np.round(distance_array(xyz[idx], xyz[idx], box=atoms.cell.cellpar()))
#    groups = np.unique(dm < dm.min()+2, axis=0)
#
#    flag = False
#    if groups.shape[0] != ngroup:
#        dm = distance_array(yy[idx].reshape(-1, 1), yy[idx].reshape(-1, 1))
#        groups = np.unique(dm < 2, axis=0) 
#        if groups.shape[0] != ngroup: 
#            raise ValueError("grouping result does not match 'ngroup'. First check if the row number match 'ngroup'. Consider move your structure")
#        else:
#            flag = True
#    else:
#        pass
#    # LAST: sort ti5c according to rows
#    def sort_row(row, dat=yy):
#        return row[np.argsort(dat[row])]
#    rows = idx[groups[0]], idx[groups[1]]
#    if flag:
#        st = partial(sort_row, dat=xx)
#    else:
#        st = partial(sort_row, dat=yy)
#    res = np.array(list(map(st, rows)))
#    return res


# tricks
def get_sym_edge(atoms, idx_l_edge4=0):
    """Translate the rutile <1-11> edge-water interface model s.t. it looks
    pretty and symmetric. 

    The trick is, when using experimental rutile structrue and ase 'surface'
    method generates the edge model, the lower surface 4-coordinated edge Ti has index 1
    (0-based). Setting this paticular atom to [0.3, 0.3, 3.5] will get the model
    look symmetric in the simulation box, thus the model becomes easier to
    handle. Here symmetric and pretty mean the model looks like:

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
    ooooooooooooooooooooooooooooo
    ooooooooooooooooooooooooooooo  [110]
    ooooooooooooooooooooooooooooo  ^ 
    **************************000  |             o: TiO2 slab
    ***************************00  |             0: <1-11>-edge
    *****************************  -----> [001]  *: water


    Args:
        atoms (ase.Atoms): 
            the edge model ASE atoms.
        idx_l_edge4 (int, optional): 
            index of lower surface edge M_4c atom. 

    Returns:
        ase.Atoms: 
            symmetric & pretty-looking edge model
    """
    target_pos = np.array([0.3, 0.3, 3.5])
    trans = target_pos - atoms.positions[idx_l_edge4]
    atoms.translate(trans)
    atoms.wrap()
    return atoms

