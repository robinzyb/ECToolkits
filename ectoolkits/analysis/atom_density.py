import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.neighborlist import neighbor_list
from MDAnalysis.analysis.base import AnalysisBase

from ectoolkits.structures.slab import Slab
from ectoolkits.utils.utils import mic_1d
from ectoolkits.log import get_logger

logger = get_logger(__name__)


def analysis_run(args):
    if args.CONFIG:
        import json
        logger.info("Analysis Start!")
        with open(args.CONFIG, 'r') as f:
            inp = json.load(f)
        system = AtomDensity(inp)
        system.run()
        logger.info("FINISHED")


class AtomDensity():
    def __init__(self, inp):
        logger.info("Perform Atom Density Analysis")
        # print file name
        self.twoD = False
        self.xyz_file = inp.get("xyz_file")
        if not os.path.isfile(self.xyz_file):
            raise FileNotFoundError

        logger.info("Read Structure File: {0}".format(inp["xyz_file"]))
        # print the cell
        self.cell = inp["cell"]
        logger.info("Read the Cell Info: {0}".format(inp["cell"]))
        # print surface 1
        self.surf1 = inp["surf1"]
        self.surf1 = np.array(self.surf1)
        logger.info("Read Surface 1 Atoms Index: {0}".format(inp["surf1"]))
        # print surface 2
        self.surf2 = inp["surf2"]
        self.surf2 = np.array(self.surf2)
        logger.info("Read Surface 2 Atoms Index: {0}".format(inp["surf2"]))

        if np.all(self.surf1 == self.surf2):
            self.twoD = True
            logger.info("Surface 1 and Surface 2 are the same, this is a 2D material.")

        self.atom_density = {}
        self.atom_density_z = {}
#        self.shift_center = inp["shift_center"]
#        logger.info("Density will shift center to water center: {0}".format(self.shift_center))

        # slice for stucture
        index = inp.get("nframe", ":")

        self.density_type = inp.get("density_type")

        # Start reading structure
        logger.info("Now Start Reading Structures")
        logger.info("----------------------------")
        self.poses = read(self.xyz_file, index=index)
        self.poses[0] = Slab(self.poses[0])
        for pos in self.poses:
            # wrap the cell
            pos.set_cell(self.cell)
            pos.set_pbc(True)
            pos.wrap()
        logger.info("Reading Structures is Finished")

        self.nframe = len(self.poses)
        logger.info("Read Frame Number: {0}".format(self.nframe))
        self.natom = len(self.poses[0])
        logger.info("Read Atom Number: {0}".format(self.natom))

    def run(self):
        # read all structure and corresponding z
        self.all_z = self.get_all_z()

        # cell info
        self.cell_volume = self.poses[0].get_volume()
        _cell = self.poses[0].get_cell()
        self.xy_area = self.cell_volume/_cell[2][2]

        # surface 1 and 2 position along the trajectory
        self.surf1_z_list = self.get_surf1_z_list()
        self.surf2_z_list = self.get_surf2_z_list()
        self.surf1_z = self.surf1_z_list.mean()
        self.surf2_z = self.surf2_z_list.mean()

        # find the water center along the trajectory
        self.water_cent_list = self.get_water_cent_list()

        # water center relative to fisrt frame
        # self.water_cent_rel_s = self.water_cent_s - self.water_cent_s[0]

        # logger.info("Calculated Origin Water Center Position: {0} A".format(self.water_cent))
        # logger.info("Water Center will shift to Cell Center: {0} A".format(self.cell[2]/2))

        for param in self.density_type:
            idx_list = self.get_idx_list(param)
            self.get_atom_density(param, idx_list=idx_list)

        self.atom_density = pd.DataFrame(self.atom_density)
        self.atom_density_z = pd.DataFrame(self.atom_density_z)
        # self.get_o_density()
        # self.dump_o_density()

    def get_all_z(self) -> np.array:
        """get the z coordinates of atoms along trajectory

        _extended_summary_

        Returns:
            np.array: the z coordinates of atoms
        """

        all_z = []
        for pos in self.poses:
            # z
            all_z.append(pos.get_positions().T[2])
        all_z = np.stack(all_z)
        return all_z

    def get_surf1_z_list(self) -> np.array:
        """calculate the surface 1 average position

        _extended_summary_

        Returns:
            np.array: axis 0: traj
        """

        surf1_z_list = self.all_z.T[self.surf1]
        surf1_z_list = surf1_z_list.T

        # wrap the surface atoms
        new_surf1_z_list = []
        for surf1_z in surf1_z_list:
            surf1_z = mic_1d(surf1_z, self.cell[2])
            new_surf1_z_list.append(surf1_z)

        surf1_z_list = np.stack(new_surf1_z_list)

        surf1_z_list = surf1_z_list.mean(axis=1)
        # all the z positions are wrapped to the first frame
        surf1_z_list = mic_1d(surf1_z_list, self.cell[2])
        return surf1_z_list

    def get_surf2_z_list(self) -> np.array:
        """calculate the surface 2 average position

        _extended_summary_

        Returns:
            np.array: axis 0: traj
        """

        surf2_z_list = self.all_z.T[self.surf2]
        surf2_z_list = surf2_z_list.T

        # wrap the surface atoms
        new_surf2_z_list = []
        for surf2_z in surf2_z_list:
            surf2_z = mic_1d(surf2_z, self.cell[2])
            new_surf2_z_list.append(surf2_z)
        surf2_z_list = np.stack(new_surf2_z_list)

        surf2_z_list = surf2_z_list.mean(axis=1)
        # all the z positions are wrapped to the first frame
        surf2_z_list = mic_1d(surf2_z_list, self.cell[2])
        return surf2_z_list

    def get_water_cent_list(self) -> np.array:
        water_cent_list = []
        for surf1_z, surf2_z in zip(self.surf1_z_list, self.surf2_z_list):
            if surf2_z < surf1_z:
                surf2_z += self.cell[2]
            water_cent = (surf1_z + surf2_z)/2
            water_cent_list.append(water_cent)
        water_cent_list = np.array(water_cent_list)
        return water_cent_list

    def water_O_idx(self):
        # guess the o index of water
        i, j = neighbor_list('ij', self.poses[0], {('O', 'H'): 1.3})
        cn = np.bincount(i)

        H2O_pair_list = []
        Ow_idx = np.where(cn == 2)[0]
        np.savetxt(os.path.join(os.path.dirname(
            self.xyz_file), "Ow_idx.dat"), Ow_idx, fmt='%d')
        return Ow_idx

    def get_idx_list(self, param):
        # get the idx from external input

        idx_method = param.get("idx_method")
        if idx_method == "manual":
            idx_list = self.get_idx_list_manual(param)
        elif idx_method == "all":
            idx_list = self.get_idx_list_all(param)
        else:
            logger.info("Not implement")
            raise ValueError
        return idx_list

    def get_idx_list_manual(self, param):
        idx_list = param.get("idx_list")
        return idx_list

    def get_idx_list_all(self, param):
        element = param.get("element")
        idx_list = self.poses[0].find_element_idx_list(element=element)
        return idx_list

    def get_atom_density(self, param, idx_list):
        logger.info("START GETTING ATOM DENSITY")
        logger.info("----------------------------")

        dz = param.get("dz", 0.05)

        atom_z = self.all_z.T[idx_list]

        atom_z = atom_z - self.surf1_z_list

        # this might cause the num exceed cell boundary, need wrap the number
        atom_z_new = []
        cell_z = self.poses[0].get_cell()[2][2]
        for num in atom_z.flatten():
            atom_z_new.append(num % cell_z)
        atom_z = np.array(atom_z_new)

        # find the length between two surface
        if (self.surf1_z > self.surf2_z) or self.twoD:
            self.surf_space = self.surf2_z + cell_z - self.surf1_z
        else:
            self.surf_space = self.surf2_z - self.surf1_z

        bins = int(self.surf_space/dz)

        density, z = np.histogram(
            atom_z, bins=bins, range=(0, self.surf_space))

        # throw the last one and move the number half bin
        z = z[:-1] + dz/2
        # z = z[:-1]

        unit_conversion = self.get_unit_conversion(param, dz)

        # normalized wrt density of bulk water
        density = density/self.nframe * unit_conversion

        element = param.get("element")
        output_file = param.get("name", f"{element}_output")

        self.atom_density_z[output_file] = z
        self.atom_density[output_file] = density

        output_file = f"{output_file}.dat"
        np.savetxt(
            output_file,
            np.stack((z, density)).T,
            header="FIELD: z[A], atom_density"
        )
        logger.info(f"Density Profile Data Save to {output_file}")

    def get_unit_conversion(self, param, dz):
        density_unit = param.get("density_unit", "number")
        if density_unit == "water":
            bulk_density = 32/9.86**3
            unit_conversion = self.xy_area*dz*bulk_density
            unit_conversion = 1.0/unit_conversion
        elif density_unit == "number":
            unit_conversion = 1.0

        return unit_conversion

    def get_ave_density(self, width_list):
        all_cent_density = {}
        all_cent_density[f"width"] = width_list
        for name, density in self.atom_density.items():
            z = self.atom_density_z[name]
            cent = z.values[-1]/2
            cent_density_list = []
            for width in width_list:
                left_bound = cent - width/2
                right_bound = cent + width/2
                part_density = density[np.logical_and(
                    (z >= left_bound), (z <= right_bound))]
                cent_density = part_density.mean()
                cent_density_list.append(cent_density)

            all_cent_density[f"{name}"] = cent_density_list

        return pd.DataFrame(all_cent_density)

    def plot_density(self, sym=False):
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
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot()
        for name, density in self.atom_density.items():
            z = self.atom_density_z[name]
            density = density.values
            if sym:
                density = (np.flip(density) + density)/2
            ax.plot(z, density, label=name)
        ax.legend()
        ax.set_ylabel("Density")
        ax.set_xlabel("z [A]")
        return fig



class Density(AnalysisBase):
    """
    Calculate the density profile of a group of atoms along the z-axis for interface systems
    """
    def __init__(self, atomgroup, **kwargs):
        super(Density, self).__init__(atomgroup.universe.trajectory,
                                          **kwargs)

        # Params
        # selection_area
        # selection_left_surface
        # selection_right_surface
        # solid |(surface left) liquid |(surface_right) solid
        # dz: default 0.05 resolution for density profile

        #self._param = param
        self.selection_area = kwargs.get("selection_area")
        self.selection_left_surface = kwargs.get("selection_left_surface")
        self.selection_right_surface = kwargs.get("selection_right_surface")
        self.dz = kwargs.get("dz", 0.05)
        self.output_file = kwargs.get("output_file", "density.dat")
        self.density_type = kwargs.get("density_type", "water")

        self._ag = atomgroup
        self.cellpar = self._ag.dimensions
        self.volume = self._ag.volume
        self.xy_area = self.volume/self.cellpar[2]

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.results.example_result = []
        self._ag_area = self._ag.select_atoms(self.selection_area)
        self._ag_left_surface = self._ag.select_atoms(self.selection_left_surface)
        self._ag_right_surface = self._ag.select_atoms(self.selection_right_surface)


        # store the histogram of z area
        self.z_area = np.zeros(self.n_frames, len(self._ag_area))
        self.z_left_surface = np.zeros(self.n_frames)
        self.z_right_surface = np.zeros(self.n_frames)


    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store an example_result of `some_function` for a single frame
        # water density at each frame
        #self.results.example_result.append(some_function(self._ag,
        #                                                self._parameter))

        # get the z coordinates of atoms along trajectory
        #self._ag_area =
        self.z_left_surface[self._frame_index] = self._ag_left_surface.positions.T[2].mean()
        self.z_right_surface[self._frame_index] = self._ag_right_surface.positions.T[2].mean()
        self.z_area[self._frame_index] = self._ag_area.positions.T[2]

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.


        self.z_area = self.z_area - self.z_left_surface

        cell_z = self.cellpar[2]
        if (self.z_left_surface > self.z_right_surface) or self.twoD:
            self.electrolyte_width = self.z_right_surface + cell_z - self.z_left_surface
        else:
            self.electrolyte_width = self.z_right_surface - self.z_left_surface

        bins = int(self.electrolyte_widthe/self.dz)

        density, z = np.histogram(
            self.z_area,
            bins=bins,
            range=(0, self.electrolyte_width)
            )

        # throw the last value and increase the half bin
        z = z[:-1] + self.dz/2

        unit_conversion = self.get_unit_conversion()

        density = density/self.n_frames * unit_conversion

        # self.atom_density_z[output_file] = z
        # self.atom_density[output_file] = density

        np.savetxt(
            self.output_file,
            np.stack((z, density)).T,
            header="FIELD: z[A], atom_density"
        )

        logger.info(f"Density Profile Data Save to {self.output_file}")

    def get_unit_conversion(self):
        density_type = self.density_type
        if density_type == "water":
            bulk_density = 32/9.86**3
            unit_conversion = self.xy_area*self.dz*bulk_density
            unit_conversion = 1.0/unit_conversion
        elif density_type == "number":
            unit_conversion = 1.0

        return unit_conversion