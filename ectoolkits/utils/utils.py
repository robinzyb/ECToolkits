"""
this script put misc function here.
"""
from ase.geometry.analysis import Analysis
import numpy as np
import os
import shutil

# frequently used unit convertion
au2eV = 27.211386245988
au2A = 0.529177210903


def get_cum_mean(array):
    tot = 0.0
    cum_mean_array = []
    for idx, i in enumerate(array):
        tot += i
        cum_mean_array.append(tot/(idx+1))
    cum_mean_array = np.array(cum_mean_array)
    return cum_mean_array


def fancy_print(string):
    print("ToolKit: {0}".format(string))


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


def printtbox(arg):
    """
    This function is a print decorated with a few characters so that the
    print function present a characteristic string. Useful for postprocess.
    """
    print("--> Toolkit: {0}".format(arg))


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
