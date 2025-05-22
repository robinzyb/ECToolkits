"""
this script put misc function here.
"""

from typing import Union
import os
import shutil
from random import random
import numbers

import numpy as np
import numpy.typing as npt
from ase.geometry.analysis import Analysis
from ase.build import molecule

from ectoolkits.structures.slab import Slab

# frequently used unit convertion
au2eV = 27.211386245988
au2A = 0.529177210903


def mic_1d(
    array: npt.NDArray[np.float64],
    cell: float,
    reference: Union[str, float] = "first",
) -> npt.NDArray[np.float64]:
    """
    Apply the minimum image convention (MIC) to a 1D array.

    This function translates positions in a one-dimensional periodic system to
    the unit cell centered around a certain reference point.
    By default, the reference point is the first element of the array.

    Parameters
    ----------
    array : numpy.ndarray
        A 1D array of float values representing positions.
    cell : float
        The length of the periodic cell.
    reference : {'first', float}, optional
        The reference point for adjustments. Use the first element of `array`
        if 'first', or a custom numeric value. Defaults to 'first'.

    Returns
    -------
    numpy.ndarray
        The adjusted array of positions within the principal cell centered
        around the reference point.
    """
    if reference == "first":
        _ref = array[0]
    elif isinstance(reference, numbers.Number):
        _ref = reference
    else:
        raise ValueError(
            "'reference' must be 'first' or a number, "
            f"got {reference} of type {type(reference)}"
        )
    _tmp_arr = array - _ref
    _tmp_arr = _tmp_arr - np.round(_tmp_arr / cell) * cell
    return _tmp_arr + _ref


def insert_water(atoms, z1, z2, water_num, model='random', space_x=0.3, space_y=0.3, space_z=0.3):
    # make a copy
    tmp = atoms.copy()
    tmp = Slab(tmp)
    point_x = int(np.linalg.norm(tmp.get_cell()[0])/space_x)
    point_y = int(np.linalg.norm(tmp.get_cell()[1])/space_y)
    point_z = int((z2-z1)/space_z)
    if water_num == 0:
        return tmp
    if model == 'grid':
        water_added_counter = 0
        for i in np.linspace(0.0, 1.0, point_x):
            for j in np.linspace(0.0, 1.0, point_y):
                for k in np.linspace(0.0, 1.0, point_z):
                    water_z = z1+(z2-z1)*k
                    cell_vec_a = tmp.get_cell()[0]
                    cell_vec_b = tmp.get_cell()[1]
                    water_xyz_pos = cell_vec_a*i + cell_vec_b*j
                    water_xyz_pos[2] += water_z
                    H2O = molecule("H2O")
                    H2O.translate(water_xyz_pos)
                    tmp.extend(H2O)
                    O_idx = len(tmp)-3
                    if len(tmp.get_neighbor_list(O_idx, 2.5)) == 2:
                        water_added_counter += 1
                        print("add water at {0}".format(water_xyz_pos))
                        print("add one water succeessfully")
                        if water_added_counter == water_num:
                            print(f"add {water_num} water molecules done")
                            return tmp
                    else:
                        del tmp[-3:]
                        print(f"reject, the water overlaps with other water at {water_xyz_pos}", end='\r')

        print(f"add {water_added_counter} water molecules, could not find enough space to add water")
        return tmp
    elif model == 'random':
        for i in range(water_num):
            water_overlap = True
            while water_overlap:
                water_z = z1+(z2-z1)*random()
                cell_vec_a = tmp.get_cell()[0]
                cell_vec_b = tmp.get_cell()[1]
                water_xyz_pos = cell_vec_a*random() + cell_vec_b*random()
                water_xyz_pos[2] += water_z
                H2O = molecule("H2O")
                H2O.translate(water_xyz_pos)
                tmp.extend(H2O)
                O_idx = len(tmp)-3
                if len(tmp.get_neighbor_list(O_idx, 2.5)) == 2:
                    water_overlap = False
                    print("add water at {0}".format(water_xyz_pos))
                    print("add one water succeessfully")
                else:
                    del tmp[-3:]
                    print("reject, the water overlaps with other water", end='\r')
        return tmp
    else:
        print("model must be grid or random")
        return None


def get_cum_mean(array):
    tot = 0.0
    cum_mean_array = []
    for idx, i in enumerate(array):
        tot += i
        cum_mean_array.append(tot/(idx+1))
    cum_mean_array = np.array(cum_mean_array)
    return cum_mean_array

def set_pbc(pos, cell):
    """set pbc for a list of Atoms object"""
    for single_pos in pos:
        single_pos.set_cell(cell)
        single_pos.set_pbc(True)


def get_rdf_list(pos, r, nbin, frames, elements):
    """
    pos: a list of atoms object
    r: the radial length
    nbin: the bin number in the radial range
    frames: how much pos number will you consider
    elements: the atom pair
    """
    tmp_info = Analysis(pos)
    # this wil get a rdf for every snapshot
    tmp_rdf_list = tmp_info.get_rdf(
        r, nbin, imageIdx=slice(0, frames, 1), elements=elements)
    return tmp_rdf_list


def get_rdf(pos, r, nbin, frames, elements):
    """

    """
    tmp_rdf_list = get_rdf_list(pos, r, nbin, frames, elements)
    tot_gr = np.zeros(nbin)
    for s_gr in tmp_rdf_list:
        tot_gr += s_gr/frames
    return tot_gr

def file_content(file, num):
    # read a specific line of file or return the block
    # file: enter file name
    # num: a integer -> return specific line content
    #      a tuple (num1, num2) -> return the line content
    #                              between num1 and num2-1
    #      a tuple (num1, ) -> return the line content from
    #                          num1, to the end of file
    if isinstance(num, int):
        with open(file) as f:
            for _idx, line in enumerate(f):
                if _idx == num:
                    return line
    elif isinstance(num, tuple):
        content = ""
        if len(num) == 2:
            with open(file) as f:
                for _idx, line in enumerate(f):
                    if (_idx >= num[0]) and (_idx < num[1]):
                        content += line
                    elif _idx >= num[1]:
                        break
                    else:
                        continue
            return content
        elif len(num) == 1:
            with open(file) as f:
                for _idx, line in enumerate(f):
                    if (_idx >= num[0]):
                        content += line
            return content
        else:
            raise ValueError("The length of range is wrong!")


def create_path(path, bk=False):
    """create 'path' directory. If 'path' already exists, then check 'bk':
       if 'bk' is True, backup original directory and create new directory naming 'path';
       if 'bk' is False, do nothing.

    Args:
        path ('str' or 'os.path'): The direcotry you are making.
        bk (bool, optional): If . Defaults to False.
    """
    path += '/'
    if os.path.isdir(path):
        if bk:
            dirname = os.path.dirname(path)
            counter = 0
            while True:
                bkdirname = dirname + ".bk{0:03d}".format(counter)
                if not os.path.isdir(bkdirname):
                    shutil.move(dirname, bkdirname)
                    break
                counter += 1
            os.makedirs(path)
            print("Target path '{0}' exsists. Backup this path to '{1}'.".format(
                path, bkdirname))
        else:
            print(
                "Target path '{0}' exsists. No backup for this path.".format(path))
    else:
        os.makedirs(path)
