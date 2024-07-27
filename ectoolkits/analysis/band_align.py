import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read, write
from cp2kdata import Cp2kCube
from cp2kdata.units import au2A, au2eV

from ectoolkits.utils.utils import get_cum_mean
from ectoolkits.log import get_logger

logger = get_logger(__name__)



#from ..utils.utils import fancy_print
# plt.style.use('./matplotlibstyle/project.mplstyle')

# inp = {
#     "input_type": "cube",
#     "ave_param":{
#         "prefix": xx,
#         "index",
#         "l1",
#         "l2":0,
#         "ncov":2,
#         "save":True,
#         "axis":'z',
#         "save_path":"."
#     }
#     "shift_param":{
#         "surf1_idx":
#         "surf2_idx":
#     }
#     "water_width_list": []
#     "solid_width_list": []

# }


class BandAlign():
    """
    Class for Band Alignment.
    only require hartree cube input.

    _extended_summary_
    """

    def __init__(self, inp: dict):
        """
        input neccesary argument.

        _extended_summary_

        Args:
            inp (dict): _description_
        """
        self.input_type = inp.get("input_type")
        logger.info("The following is input you have")
        print(inp)

        if self.input_type == "cube":
            self.pav_x_list, self.pav_list, self.mav_x_list, self.mav_list, self.traj = \
                self.get_pav_mav_traj_list_from_cube(**inp.get("ave_param"))
        elif self.input_type == "file":
            self.pav_x_list, self.pav_list, self.mav_x_list, self.mav_list, self.traj = \
                self.get_pav_mav_traj_list_from_file(
                    inp.get("ave_param").get("save_path"))

        self.surf1_idx = inp.get("shift_param").get("surf1_idx")
        self.surf2_idx = inp.get("shift_param").get("surf2_idx")
        self.water_width_list = inp.get("water_width_list")
        self.solid_width_list = inp.get("solid_width_list")
        self.water_cent_list, self.solid_cent_list = self.get_cent_list()
        self.water_hartree_list = self.get_water_hartree()
        self.solid_hartree_list = self.get_solid_hartree()

    def plot_hartree_per_width(self, part='solid'):
        if part == 'solid':
            hartree_list = self.solid_hartree_list
        elif part == 'water':
            hartree_list = self.water_hartree_list
        plt.rc('font', size=18)  # controls default text size
        plt.rc('axes', titlesize=23)  # fontsize of the title
        plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=18)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=18)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=16)  # fontsize of the legend

        # controls default text size
        plt.rc('lines', linewidth=2, markersize=10)

        plt.rc('axes', linewidth=2)
        plt.rc('xtick.major', size=10, width=2)
        plt.rc('ytick.major', size=10, width=2)
        num_width = len(hartree_list.columns)
        num_row = int(num_width/2) + 2
        num_col = 2

        fig = plt.figure(figsize=(16, 4.5*num_row), dpi=200)
        gs = fig.add_gridspec(num_row, num_col)

        # plot cum lines
        ax0 = fig.add_subplot(gs[0])
        for width, hartree_list_per_width in hartree_list.items():
            ax0.plot(get_cum_mean(hartree_list_per_width),
                     label=f"width {width}")
        ax0.legend(ncol=2)
        ax0.set_xlabel("Frame Index")
        ax0.set_ylabel("Hartree [eV]")
        ax0.set_title("Cumulative Hartree")

        # plot mean alignwidth
        ax1 = fig.add_subplot(gs[1])
        mean_serial = hartree_list.mean()
        ax1.plot(mean_serial, '-o', markerfacecolor='white',
                 markeredgecolor='black')
        ax1.set_xlabel("Width [Å]")
        ax1.set_ylabel("Hartree [eV]")
        ax1.set_title("Last Mean Hartree")

        # plot cum lines
        for idx, (width, hartree_list_per_width) in enumerate(hartree_list.items()):
            tmp_ax = fig.add_subplot(gs[idx+2])
            tmp_ax.plot(hartree_list_per_width, label=f"width {width}")
            tmp_ax.set_xlabel("Frame Index")
            tmp_ax.set_ylabel("Hartree [eV]")
            tmp_ax.set_title(f"Width {width}")
        fig.tight_layout()
        # fig.show()
        return fig

    def get_water_hartree(self) -> pd.DataFrame:
        """Obtain hartree from water region in interface model.


        _extended_summary_

        Returns:
            pd.DataFrame: return the hartree_list as pandas DataFrame
        """
        hartree_list = {}
        for width in self.water_width_list:
            hartree_list_per_width = []
            for x, pav, water_cent, snapshot in zip(self.pav_x_list, self.pav_list, self.water_cent_list, self.traj):
                cell_z = snapshot.get_cell()[2][2]
                water_pav = pav[get_range_bool(x, water_cent, width, cell_z)]
                hartree_list_per_width.append(water_pav.mean())

            hartree_list[f"{width}"] = np.array(hartree_list_per_width)
        water_hartree_list = pd.DataFrame(hartree_list)
        return water_hartree_list

    def get_solid_hartree(self):
        hartree_list = {}
        for width in self.solid_width_list:
            hartree_list_per_width = []
            counter = 0
            for x, mav, solid_cent, snapshot in zip(self.mav_x_list, self.mav_list, self.solid_cent_list, self.traj):
                cell_z = snapshot.get_cell()[2][2]
                solid_mav = mav[get_range_bool(x, solid_cent, width, cell_z)]
                hartree_list_per_width.append(solid_mav.mean())

            hartree_list[f"{width}"] = np.array(hartree_list_per_width)
        hartree_list = pd.DataFrame(hartree_list)
        return hartree_list

    def get_cent_list(self):
        # not recommend for surface atoms shift at boundary
        water_cent_list = []
        solid_cent_list = []
        for snapshot in self.traj:
            surf1_z = get_z_mean(snapshot, self.surf1_idx)
            surf2_z = get_z_mean(snapshot, self.surf2_idx)
            cell_z = snapshot.get_cell()[2][2]
            if surf2_z < surf1_z:
                surf2_z += cell_z
            water_cent = (surf1_z + surf2_z)/2
            water_cent_list.append(water_cent)
            solid_cent_list.append(water_cent-cell_z/2)
        water_cent_list = np.array(water_cent_list)
        return water_cent_list, solid_cent_list

    def get_pav_mav_traj_list_from_file(self, save_path):
        pav_x_list = np.loadtxt(os.path.join(save_path, "pav_x_list.dat"))
        pav_list = np.loadtxt(os.path.join(save_path, "pav_list.dat"))
        mav_x_list = np.loadtxt(os.path.join(save_path, "mav_x_list.dat"))
        mav_list = np.loadtxt(os.path.join(save_path, "mav_list.dat"))
        traj = read(os.path.join(save_path, "cube_traj.xyz"), index=":")
        return pav_x_list, pav_list, mav_x_list, mav_list, traj

    def get_pav_mav_traj_list_from_cube(self, prefix, index, l1, l2=0, ncov=2, save=True, axis='z', save_path="."):
        # the input l1 l2 is in bohr
        l1 = l1/au2A
        l2 = l2/au2A

        pav_x_list = []
        pav_list = []
        mav_x_list = []
        mav_list = []
        traj = []
        for idx in range(*index):
            cube = Cp2kCube(f"{prefix}{idx}.cube")

            x, pav = cube.get_pav(interpolate=True)
            pav_x_list.append(x)
            pav_list.append(pav)

            x, mav = cube.get_mav(l1=l1, l2=l2, ncov=ncov, interpolate=True)
            mav_x_list.append(x)
            mav_list.append(mav)

            stc = cube.get_stc()
            traj.append(stc)
            print(f"process cube {idx} finished", end="\r")

        pav_x_list = np.array(pav_x_list)*au2A
        pav_list = np.array(pav_list)*au2eV
        mav_x_list = np.array(mav_x_list)*au2A
        mav_list = np.array(mav_list)*au2eV

        if save:
            np.savetxt(os.path.join(save_path, "pav_x_list.dat"),
                       pav_x_list, fmt="%3.4f")
            np.savetxt(os.path.join(save_path, "pav_list.dat"),
                       pav_list, fmt="%3.4f")
            np.savetxt(os.path.join(save_path, "mav_x_list.dat"),
                       mav_x_list, fmt="%3.4f")
            np.savetxt(os.path.join(save_path, "mav_list.dat"),
                       mav_list, fmt="%3.4f")
            write(os.path.join(save_path, "cube_traj.xyz"), traj)

        return pav_x_list, pav_list, mav_x_list, mav_list, traj


def get_range_bool(x, cent, width, cell_z):
    left_bound = cent-width/2
    left_bound = left_bound % cell_z
    right_bound = cent+width/2
    right_bound = right_bound % cell_z

    if left_bound < right_bound:
        range_bool = np.logical_and((x <= right_bound), (x >= left_bound))
    else:
        range_bool = np.logical_or((x <= right_bound), (x >= left_bound))
    return range_bool


def get_nearest_idx(array, value):
    idx = np.argmin(np.abs(array-value))
    return idx


def get_z_mean(atoms, idx_list):
    # check_in_plane: switch to true to check the selected atoms are in similar z positions
    # sometimes z postions will shift by pbc
    z_mean = atoms[idx_list].get_positions().T[2].mean()
    return z_mean


def get_slab_cent(traj, surf1_idx, surf2_idx, cell_z):
    slab_cent_list = []
    for snapshot in traj:
        surf1_z = get_z_mean(snapshot, surf1_idx)
        surf2_z = get_z_mean(snapshot, surf2_idx)
        if surf1_z < surf2_z:
            surf2_z -= cell_z
        slab_cent = (surf1_z + surf2_z)/2
        slab_cent_list.append(slab_cent)
    slab_cent_list = np.array(slab_cent_list)
    return slab_cent_list


def align_to_slab_cent(x_list, pav_list, traj, surf1_idx, surf2_idx, cell_z):
    # surf1 is solid on the left
    # surf2 is solid on the right
    # ok for pav_list and mav_list
    slab_cent_list = get_slab_cent(traj, surf1_idx, surf2_idx, cell_z)

    slab_cent_1st = slab_cent_list[0]
    slab_cent_1st_idx = get_nearest_idx(x_list[0], slab_cent_1st)
    new_pav_list = []

    for slab_cent, x, pav in zip(slab_cent_list, x_list, pav_list):
        slab_cent_idx = get_nearest_idx(x, slab_cent)
        roll_num = slab_cent_idx - slab_cent_1st_idx
        new_pav = np.roll(pav, -roll_num)
        new_pav_list.append(new_pav)
    new_pav_list = np.array(new_pav_list)
    return new_pav_list


def get_alignment_water(level, ref_water_hartree):
    U = -level + ref_water_hartree + 15.35 - 15.81 - 0.35
    return U


def get_alignment_water_2(level, ref_water_hartree, ref_solid_hartree):
    U = -level - ref_solid_hartree + ref_water_hartree + 15.35 - 15.81 - 0.35
    return U


def get_alignment_vac(level, ref_vac_hartree):
    U = -level + ref_vac_hartree - 4.44
    return U


def get_alignment_vac_2(level, ref_vac_hartree, ref_solid_hartree):
    U = -level - ref_solid_hartree + ref_vac_hartree - 4.44
    return U


def get_alignment(level, ref_hartree, ref_solid_hartree=None, vac_model=False, ref_bulk=False):
    if vac_model:
        if ref_bulk:
            U = get_alignment_vac_2(
                level=level, ref_vac_hartree=ref_hartree, ref_solid_hartree=ref_solid_hartree)
        else:
            U = get_alignment_vac(level=level, ref_vac_hartree=ref_hartree)
    else:
        if ref_bulk:
            U = get_alignment_water_2(
                level=level, ref_water_hartree=ref_hartree, ref_solid_hartree=ref_solid_hartree)
        else:
            U = get_alignment_water(level=level, ref_water_hartree=ref_hartree)
    return U


def get_z_mean(atoms, idx_list):
    z_mean = atoms[idx_list].get_positions().T[2].mean()
    return z_mean


def get_water_center_list(traj, surf1_idx, surf2_idx, cell_z):
    # not recommend for surface atoms shift at boundary
    water_center_list = []
    for snapshot in traj:
        surf1_z = get_z_mean(snapshot, surf1_idx)
        surf2_z = get_z_mean(snapshot, surf2_idx)
        if surf2_z < surf1_z:
            surf2_z += cell_z
        water_center = (surf1_z + surf2_z)/2
        water_center_list.append(water_center)
    water_center_list = np.array(water_center_list)
    return water_center_list


def get_water_hartree(x_list, pav_list, water_center_list, width_list):
    hartree_list = {}
    for width in width_list:
        hartree_list_per_width = []
        for x, pav, water_center in zip(x_list, pav_list, water_center_list):
            water_pav = pav[np.logical_and(
                (x > water_center-width/2), (x < water_center+width/2))]
            hartree_list_per_width.append(water_pav.mean())

        hartree_list[f"{width}"] = np.array(hartree_list_per_width)
    hartree_list = pd.DataFrame(hartree_list)
    return hartree_list


def get_layer_space_list(traj, layer1_idx, layer2_idx):
    layer1_z_list = []
    layer2_z_list = []
    for snapshot in traj:
        layer1_z = get_z_mean(snapshot, layer1_idx)
        layer2_z = get_z_mean(snapshot, layer2_idx)
        layer1_z_list.append(layer1_z)
        layer2_z_list.append(layer2_z)
    layer1_z_list = np.array(layer1_z_list)
    layer2_z_list = np.array(layer2_z_list)
    layer_space_list = layer2_z_list - layer1_z_list

    return layer_space_list


def get_solid_hartree(x_list, mav_list, slab_center_list, width_list):
    hartree_list = {}
    for width in width_list:
        hartree_list_per_width = []
        for x, mav, slab_center in zip(x_list, mav_list, slab_center_list):
            solid_mav = mav[np.logical_and(
                (x > slab_center-width/2), (x < slab_center+width/2))]
            hartree_list_per_width.append(solid_mav.mean())

        hartree_list[f"{width}"] = np.array(hartree_list_per_width)
    hartree_list = pd.DataFrame(hartree_list)
    return hartree_list
