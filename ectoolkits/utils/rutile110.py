# author:  Rui-Hao Bi (biruihao@westlake.edu.cn)
# summary: rutile (110)-water interface toolkit
# summary: this contains the general methods for rutile-water(110)
# 2021-2024

# general modules
import os
import glob
import numpy as np
from ase.geometry import cellpar_to_cell
from ase import Atoms
# distance caculation library functions
from MDAnalysis.lib.distances import *
from MDAnalysis.lib.c_distances import _minimize_vectors_triclinic

def get_pair(xyz, idx1, idx2, cutoff_hi, cutoff_lo=None, cell=None, **kwargs):

    """
    search possible pairs between idx1 and idx2,
    whose distance between are smaller than cutoff_hi
    """

    atoms1 = xyz[idx1]
    atoms2 = xyz[idx2]
    pairs, distances = capped_distance(reference=atoms1, configuration=atoms2,
                                   max_cutoff=cutoff_hi, min_cutoff=cutoff_lo,
                                   box=cell)
    return np.vstack((idx1[pairs[:, 0]], idx2[pairs[:, 1]])).T



def minimize_vectors_triclinic(v: np.ndarray, box: np.ndarray, ):
    """
    computes the mic vector
    """
    res = np.zeros_like(v, dtype=np.float32)
    v_32 = v.astype(np.float32)
    box_32 = box.flatten().astype(np.float32)
    _minimize_vectors_triclinic(vectors=v_32, box=box_32, output=res)

    return res

def normalized_vector(v: np.ndarray):
    """calculate the normalized vector

    Args:
        v (np.ndarray): a vector

    Returns:
        np.ndarray : normalized vector
    """
    return v / np.linalg.norm(v)

def d_unique_vecs(vecs):
    """get the unique distance vectors

    Args:
        vecs (np.ndarray): distance vectors

    Returns:
        dict: a dictionary holds the unique distance vectors.
    """
    u = {}
    count = 1
    for vec in vecs:
        flag = 0
        if len(u.keys()) == 0:
            u.update({count: [vec]})
        else:
            for k in u.keys():
                if np.linalg.norm(np.cross(vec, u[k][0])) > 0.5:
                    flag += 1
                else:
                    u[k].append(vec)
            if len(u.keys()) == flag:
                count += 1
                u.update({count: [vec]})
    return u

def g_unique_vecs(d: dict):
    """ensures the unique vectors calculated by `d_unique_vecs`.

    Handles the case when the each key in the `d` dictonary has multiple numerically close values. Take the average of these vectors as the "bond" vector.

    Args:
        d (dict): dictionary calculated from `d_unique_vecs`

    Returns:
        np.ndarray: a numpy array holds the "bond" vectors. For an octahedral center metal, there are 6 distinct vectors.
    """
    for k in d.keys():
        tmp = np.array(d[k])
        tmp = np.where((tmp[:, -1]>0)[:, np.newaxis], tmp, -tmp)
        d[k] = tmp.mean(axis=0)
    vecs = np.empty((len(d), 3), dtype=float)
    count = 0
    for k in d.keys():
        vecs[count] = d[k]
        count+=1
    return vecs

def get_octahedral_bonds(tio2_cut: Atoms, octahedral_bonds_upper_bound: float=2.4):
    """calculate the octahedral bonds (vectors) in an \\hkl(1-11) edge model

    Args:
        tio2_cut (Atoms): the edge model cut from the bulk structure. (the output of `cut_edge_from_bulk`.)
        octahedral_bonds_upper_bound (float, optional): The longest length to recogonize an oxygen as a octahedral ligand. Defaults to 2.4.

    Returns:
        np.ndarray: a coordinates array for the octahedral vectors.
    """

    # compute the distance matrix under PBC
    xyz = tio2_cut.positions
    o_idx = np.where(tio2_cut.symbols=="O")[0]
    ti_idx = np.where(tio2_cut.symbols=="Ti")[0]

    pair = get_pair(xyz, ti_idx, o_idx, cutoff_hi=2.4, cutoff_lo=None, cell=tio2_cut.cell.cellpar())
    diff = xyz[pair[:, 1]] - xyz[pair[:, 0]]
    vecs = minimize_vectors_triclinic(diff, tio2_cut.get_cell().array)

    uniqe_dist_vec = d_unique_vecs(vecs)
    octahedral_bonds = g_unique_vecs(uniqe_dist_vec)
    return octahedral_bonds

def get_rotM_edged_rutile110(tio2: Atoms, octahedral_bonds_upper_bound: float=2.4, bridge_along="x"):
    """compute the unit xyz vectors for a slab of \\hkl(1-11) edged tio2.

    !!!!! TODO ---> This method is currently very adhoc and prone to failure
    !!!!! WE NEED TO IMPROVE THIS

    Detailed algorithm:
    First, one will get all the 6-"octahedral bonds", (as detailed in `get_octahedral_bonds`)
    Second, notice the two of the enlogated octahedral bonds roughly aligns with:
        - unit z direction <the direction of the terminal water -> Ti bond >
        - unit x direction <perpendicular to the bridge water row>
    With unit z and unit x, one can reconstruct the unit x by simply cross product

    Args:
        tio2_cut (Atoms): any tio2 \\hkl(1-11) edge model (hopefully it will work for all input)
        octahedral_bonds_upper_bound (float, optional): The longest length to recogonize an oxygen as a octahedral ligand. Defaults to 2.4.
        bridge_along (str, optional): The direction of the bridge oxygens, could be either "x" or "y". Defaults to "y".

    Returns:
        np.array: the normal vectors, which could be used as the rotM matrix for the coordinates
    """

    def is_v_dominant(v, ind):
        abs_val = np.abs(v)
        if np.max(abs_val) == abs_val[ind]:
            i = np.arange(3)
            iother = i[i!=ind]
            if (abs_val[iother][0] / abs_val[ind] < 0.5) and (abs_val[iother][1] / abs_val[ind] < 0.5) and (abs_val[ind] > 1.1):
                return True
        return False

    def do_average(v_list, ind):
        abs_val = np.abs(v_list)
        avg_amplitude = abs_val.mean(axis=0)
        i = np.arange(3)
        iother = i[i!=ind]
        arg_second_largest = np.argmax(avg_amplitude[iother])
        values = v_list[:, iother[arg_second_largest]]
        signs = np.sign(values)
        if len(signs)==1:
            return v_list
        # Check if all signs are the same
        if (signs[1:] == signs[:-1]).all():
            return np.mean(v_list, axis=0)
        else:
            # Find the majority sign
            positive_count = np.sum(signs > 0)
            negative_count = np.sum(signs < 0)

            if positive_count >= negative_count:
                # If positive signs are the majority, average the positive vectors
                positive_vectors = v_list[signs > 0]
                return np.mean(positive_vectors, axis=0) if len(positive_vectors) > 2 else positive_vectors
            else:
                # If negative signs are the majority or if counts are equal, average the negative vectors
                negative_vectors = v_list[signs < 0]
                return np.mean(negative_vectors, axis=0) if len(negative_vectors) > 2 else negative_vectors

    def get_averaged(octahedral_bonds, ind):
        v_list = []
        for v in octahedral_bonds:
            if is_v_dominant(v, ind):
                out = v if v[ind] > 0 else -v
                v_list.append(out)
        v_list = np.array(v_list)
        return do_average(v_list, ind)


    octahedral_bonds = get_octahedral_bonds(tio2, octahedral_bonds_upper_bound)

    if bridge_along == "y":
        x_up = normalized_vector(get_averaged(octahedral_bonds, 0))
        z_up = normalized_vector(get_averaged(octahedral_bonds, 2))
        y_up = np.cross(z_up, x_up)
    elif bridge_along== "x":
        y_up = normalized_vector(get_averaged(octahedral_bonds, 1))
        z_up = normalized_vector(get_averaged(octahedral_bonds, 2))
        x_up = np.cross(y_up, z_up)

    return np.array([x_up, y_up, z_up])

def find_cn_idx(atoms1, atoms2, cutoff_hi, cutoff_lo=None, cell=None, **kwargs):
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
    # to avoid double counting
    cn_idx = np.unique(pairs[:, 1])
    return cn_idx


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
    # cell = np.array(cell).astype(np.float32)

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


def get_watOidx(atoms, M="Ti", d_OH_cutoff=1.2, d_MO_cutoff=2.8, cn_M_cutoff=1):
    """gets all the water oxygen indices in rutile (110)-water interface

    Args:
        atoms (ase.Atoms):
            ASE atoms. One snapshot of rutile (110)-water structure.
        M (str, optional):
            The metal atom in rutile structrue. Defaults to "Ti".

    Returns:
        watOidx (numpy.ndarray): 0-based indices for water oxygen atoms.
        watHidx (numpy.ndarray): 0-based indices for water hydrogen atoms. (all the hydrogens)
    """
    xyz = atoms.positions
    cell = atoms.cell.cellpar()
    idx_M = np.where(atoms.symbols == M)[0]
    idx_O = np.where(atoms.symbols == 'O')[0]
    idx_H = np.where(atoms.symbols == 'H')[0]

    cn_H = count_cn(xyz[idx_O, :], xyz[idx_H, :], d_OH_cutoff, None, cell)
    cn_M = count_cn(xyz[idx_O, :], xyz[idx_M, :], d_MO_cutoff, None, cell)

    watOidx = idx_O[(cn_H >= 0) * (cn_M <= cn_M_cutoff)]
    hcn_idx = find_cn_idx(xyz[watOidx, :], xyz[idx_H, :], d_OH_cutoff, None, cell)
    watHidx = idx_H[hcn_idx]
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
            The indices for the slab model.
        atoms_slab(ase.atoms):
            Slab model atoms object.
    """
    indices = np.arange(atoms.get_global_number_of_atoms())
    idx_ow, idx_hw = get_watOidx(atoms, M="Ti")
    idx_wat = np.append(idx_ow, idx_hw)
    idx_slab = np.setdiff1d(indices, idx_wat)
    atoms_slab = atoms[idx_slab]
    return idx_slab, atoms_slab


def get_rotM(vecy, vecz):
    r"""get the rotation matrix for rotating a rutile model. Specifically, rotating a step
    edge model s.t x is parrallel to [001], y is parrallel to [1-10], and z is parallel
    to \hkl<110>.

    reference:
      https://www.cnblogs.com/armme/p/10596697.html#:~:text=旋转坐标系的方法又有两种：%20Proper%20Euler%20angles%2C%20第一次与第三次旋转相同的坐标轴（z-x-z%2Cx-y-x%2C%20y-z-y%2Cz-y-z%2C%20x-z-x%2C,y-x-y）%E3%80%82%20Tait–Bryan%20angles%2C%20依次旋转三个不同的坐标轴（x-y-z%2Cy-z-x%2C%20z-x-y%2Cx-z-y%2C%20z-y-x%2C%20y-x-z）；
      or see https://shorturl.at/pvxyE

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
    vece1, vece2, vece3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]

    refj = [vecx, vecy, vecz]
    refi = [vece1, vece2, vece3]
    rotM = np.empty((3, 3), dtype=float)
    rotM[:] = np.nan

    def get_cos(vec1, vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    for ii in range(3):
        for jj in range(3):
            rotM[ii, jj] = get_cos(refj[ii], refi[jj])
    return rotM


def sep_upper_lower(z, indices):
    """given indices, seperate them to upper and lower. More specifically, from
    [<indices>] to [[<idx_upper>], [<idx_lower>]]

    Args:
        z (numpy.ndarray):
            z coordinates
        indices (numpy.ndarray):
            0-based indices

    Returns:
        numpy.ndarray: [[<idx_upper>], [<idx_lower>]]
    """
    z = z[indices]
    zmean = z.mean()
    i1 = indices[z > zmean]
    i2 = indices[z < zmean]
    return np.array([i1, i2])


def pair_M5c_n_obr(atoms, idx_cn5, idx_obrs, M="Ti"):
    """This methods will find the Ti5c's neighboring Obr atoms.

    Args:
        atoms (ASE.Atoms):
            ase atoms of your interface model.
        idx_cn5 (np.ndarray):
            1-d integer array, which contains indices for Ti5c atoms.
        idx_obrs (np.ndarray):
            1-d integer array, which contains indices for all the Obr atoms.
        M (str, optional):
            Metal elements in the rutile structure. Defaults to "Ti".

    Returns:
        tuple: (<idx_cn5>, <res_obr>).
            Paired Ti5c and Obr indices. Check if they are really paired before you use.
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
    # apply minimum image convention
    mic_v = minimize_vectors(v, box=atoms.cell.cellpar())
    sel = mic_v[:, 1] < 0                           # selective flipping
    res_idx[sel] = np.flip(res_idx[sel], axis=1)
    return idx_cn5, res_idx


# def group_by_row(atoms, idx, ngroup=2):
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
# def sort_by_row(atoms, idx, ngroup=2):
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
    target_pos = np.array([1.5, 1.5, 3.0])
    trans = target_pos - atoms.positions[idx_l_edge4]
    atoms.translate(trans)
    atoms.wrap()
    return atoms
