import os
import glob

import numpy as np
from ase.io import vasp

from ectoolkits.log import get_logger

logger = get_logger(__name__)

def scale_iso_cell(atoms, start, end, step, out_name):
    """
    creat the scaling cell for vasp
    atoms: the Atoms object from ase
    start: starting fraction
    end: ending fraction
    step: the step from start to end
    out_name: the name for out file, it can be a path.
    """
    cell_list = np.arange(start, end, step)
    old_cell_vector = atoms.get_cell()
    new_atoms = atoms.copy()
    for scale in cell_list:
        new_atoms.set_cell(old_cell_vector * scale, scale_atoms=True)
        vasp.write_vasp(out_name + "_{0:.3f}".format(scale), new_atoms,
                        direct=True, sort=True)
        logger.info("create POSCAR for {0:.3f} scaling".format(scale))


def find_outcar(dirpath, filename):
    """
    find the outcar path with some fancy output
    """
    dirpath = os.path.abspath(dirpath)
    file = os.path.join(dirpath, filename)
    allfile = glob.glob(file)
    for i in allfile:
        logger.info("The file {0} has been found".format(os.path.basename(i)))
    return allfile


def exout_vasp(files):
    """
    extract the vasp file from filepath
    """
    infos = []
    for file in files:
        logger.info("Now extract the cell parameter, volume and "
                  "energy from {0}".format(os.path.basename(file)))
        info = []
        pos = vasp.read_vasp_out(file)
        for i in pos.get_cell_lengths_and_angles():
            info.append(i)
        info.append(pos.get_volume())
        info.append(pos.get_total_energy())
        infos.append(info)
        logger.info("extraction finished")
    infos = np.array(infos)

    # sort the row by volume
    infos = infos[infos[:, 6].argsort()]
    dirname = os.path.dirname(files[0])
    info_file = os.path.join(dirname, "v-e.dat")

    np.savetxt(info_file, infos, fmt='%.8f',
               header="lengthA  lengthB  lengthC  AngleA  AngleB  AngleC"
               "Volume Energy")
    logger.info("store the file in {0}".format(info_file))
    return infos
