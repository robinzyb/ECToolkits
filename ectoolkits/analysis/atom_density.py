import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.cell import Cell # used for converting cell parameters.
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.mdamath import box_volume

import matplotlib as mpl
import cp2kdata.plots.colormaps

from ectoolkits.structures.slab import Slab
from ectoolkits.utils.utils import mic_1d
from ectoolkits.log import get_logger

logger = get_logger(__name__)

def run_atom_density_analysis(inp):
    xyz_file = inp.get("xyz_file")
    format = inp.get("format", "XYZ")
    cell = inp.get("cell")
    cell = Cell.new(cell)
    u = Universe(xyz_file, format=format)
    u.atoms.dimensions = cell.cellpar()
    surf1 = inp.get("surf1")
    surf2 = inp.get("surf2")
    density_type = inp.get("density_type")

    ad = AtomDensity(atomgroup=u.atoms,
            surf1=surf1,
            surf2=surf2,
            density_type=density_type)
    ad.run()
    return ad

class AtomDensity(AnalysisBase):
    """
    Class for analyzing atom density profiles along the z-axis for interface systems.

    This class extends the MDAnalysis `AnalysisBase` class and provides methods to compute and analyze
    atom density distributions along the z-axis, which is useful for studying interfaces and layered materials.

    Attributes:
        twoD (bool): Whether the system is a 2D material (surface 1 and 2 are the same).
        surf1 (np.ndarray): Indices of atoms defining surface 1 (left interface).
        surf2 (np.ndarray): Indices of atoms defining surface 2 (right interface).
        density_type (Any): Parameters specifying which atom types to analyze.
        ag (AtomGroup): AtomGroup of all atoms to be analyzed.
        u (Universe): The MDAnalysis Universe object containing the atomgroup and trajectory.
        cellpar (np.ndarray): Unit cell parameters of the system.
        n_atoms (int): Number of atoms in the selected atom group.
        volume (float): Volume of the simulation cell.
        xy_area (float): Area of the xy-plane of the simulation cell.
        n_results (int): Number of results to store for each frame.
        all_z (np.ndarray): Array to store z-coordinates of all atoms for each frame.
        surf1_z_list (np.ndarray): Average z positions of surface 1 atoms for each frame.
        surf2_z_list (np.ndarray): Average z positions of surface 2 atoms for each frame.
        surf1_z (float): Mean z position of surface 1.
        surf2_z (float): Mean z position of surface 2.
        water_cent_list (np.ndarray): List of water center positions along the trajectory.
        atom_density (dict): Dictionary to store computed atom density profiles.
        atom_density_z (dict): Dictionary to store z-coordinates corresponding to density profiles.

    Methods:
        _prepare(): Prepares data structures before trajectory iteration.
        _single_frame(): Processes a single frame to extract z-coordinates.
        _conclude(): Finalizes the analysis and computes density profiles.
        get_surf_z_list(idxs_surf): Calculates average z positions for a given surface.
        get_water_cent_list(): Computes the water center positions along the trajectory.
        get_idx_list(param): Determines atom indices to analyze based on input parameters.
        get_idx_list_manual(param): Returns manually specified atom indices.
        get_idx_list_all(param): Returns all atom indices of a specified element.
        get_atom_density(param, idx_list): Calculates and saves the atom density profile.
        get_unit_conversion(density_unit, dz, xy_area): Computes the unit conversion factor for density.
    """
    def __init__(self,
                 atomgroup,
                 verbose=True,
                 **kwargs):
        trajectory = atomgroup.universe.trajectory
        super(AtomDensity, self).__init__(trajectory,
                                      verbose=verbose)
        """
        Initialize the Density analysis.

        Parameters:
            atomgroup (AtomGroup): The atom group to analyze.
            verbose (bool): Whether to print verbose output. Defaults to True.
            **kwargs: Additional keyword arguments, including:
                twoD (bool): Whether the system is a 2D material (surface 1 and 2 are the same).
                surf1 (List[int]): Indices of atoms defining surface 1 (left interface).
                surf2 (List[int]): Indices of atoms defining surface 2 (right interface).
                density_type (Dict): Parameters specifying which atom types to analyze.

        Sets up the initial analysis parameters, parses input surfaces and density types,
        and initializes attributes for cell, atom group, and logging.
        """
        # Params
        # solid |(surface left) liquid |(surface_right) solid
        # dz: default 0.05 resolution for density profile


        logger.info("Analysis of Atom Density along Z axis")
        logger.info("Only trajectory from NVT ensemble is supported")

        # Parse external input parameters
        self.twoD = kwargs.get("twoD", False)
        # better to change the names of surf1 and surf2 to left and right
        self.surf1 = kwargs.get("surf1", None)
        logger.info("Read Surface 1 Atoms Index: {0}".format(self.surf1))
        self.surf1 = np.array(self.surf1)
        self.surf2 = kwargs.get("surf2", None)
        logger.info("Read Surface 2 Atoms Index: {0}".format(self.surf2))
        self.surf2 = np.array(self.surf2)
        if np.all(self.surf1 == self.surf2):
            self.twoD = True
            logger.info("Surface 1 and Surface 2 are the same, this is a 2D material.")
        self.density_type = kwargs.get("density_type")

        #TODO: MDA cellpar only accepts [a, b, c, alpha, beta, gamma] format
        # use ase cells to covert to this format.
        # ensure the dimensions are not empty.
        # internal variable
        self.ag = atomgroup
        self.u = self.ag.universe
        self.cellpar = self.ag.dimensions
        self.n_atoms = len(self.ag)
        self.volume = box_volume(self.cellpar)
        self.xy_area = self.volume/self.cellpar[2]
        self.atom_density = {}
        self.atom_density_z = {}

        logger.info("Read Frame Number: {0}".format(self.u.trajectory.n_frames))
        logger.info("Read Atom Number: {0}".format(self.n_atoms))
        logger.info("Read Cell Parameters: {0}".format(self.cellpar))
        logger.info("Read z length: {0}".format(self.cellpar[2]))
        logger.info("Read XY Area: {0}".format(self.xy_area))
        logger.info("Read Cell Volume: {0}".format(self.volume))



        self.n_results = 2

    def _prepare(self):
        """
        Prepare data structures before trajectory iteration.

        Initializes the array to store z-coordinates of all atoms for each frame.
        Called before the trajectory iteration begins.
        """
        self.all_z = np.zeros((self.n_frames, self.n_atoms))

    def _single_frame(self):
        """
        Process a single frame to extract z-coordinates.

        Stores the z-coordinates of all atoms in the current frame into the results array.
        """
        #TODO: all_z seems not necessary, since we can get the z from u.atoms.positions
        self.all_z[self._frame_index] = self.ag.positions[:, 2]

    def _conclude(self):
        """
        Finalize the analysis and compute density profiles.

        Calculates average surface positions, water center positions, and computes
        atom density profiles for each specified atom type.
        """
        self.surf1_z_list = self.get_surf_z_list(idxs_surf=self.surf1)
        self.surf2_z_list = self.get_surf_z_list(idxs_surf=self.surf2)
        self.surf1_z = self.surf1_z_list.mean()
        self.surf2_z = self.surf2_z_list.mean()
        # find the water center along the trajectory
        self.water_cent_list = self.get_water_cent_list()

        for param in self.density_type:
            idx_list = self.get_idx_list(param)
            self.get_atom_density(param, idx_list=idx_list)


    def get_surf_z_list(self, idxs_surf: np.array) -> np.array:
        """
        Calculate the average z positions for a given surface.

        Wraps the z-coordinates of surface atoms to handle periodic boundaries and computes
        the mean z position for each frame.

        Args:
            idxs_surf (np.array): Indices of surface atoms.

        Returns:
            np.array: Average z positions for each frame.
        """

        surf_z_list = self.all_z.T[idxs_surf]
        surf_z_list = surf_z_list.T

        # Wrap the surface atoms.
        # Sometimes atoms on surfaces are distributed around the cell boundary.
        # The absolute positions of certain atoms may differ by cell length.
        # This will cause the average position of surface atoms to be wrong.
        new_surf_z_list = []
        for surf_z in surf_z_list:
            surf_z = mic_1d(surf_z, self.cellpar[2])
            new_surf_z_list.append(surf_z)

        surf_z_list = np.stack(new_surf_z_list)

        surf_z_list = surf_z_list.mean(axis=1)
        # all the z positions are wrapped to the first frame
        surf_z_list = mic_1d(surf_z_list, self.cellpar[2])
        return surf_z_list

    def get_water_cent_list(self) -> np.array:
        """
        Compute the water center positions along the trajectory.

        Calculates the midpoint between the two surfaces for each frame.

        Returns:
            np.array: Water center positions for each frame.
        """
        water_cent_list = []
        for surf1_z, surf2_z in zip(self.surf1_z_list, self.surf2_z_list):
            if surf2_z < surf1_z:
                surf2_z += self.cellpar[2]
            water_cent = (surf1_z + surf2_z)/2
            water_cent_list.append(water_cent)
        water_cent_list = np.array(water_cent_list)
        return water_cent_list

    def get_idx_list(self, param):
        """
        Determine atom indices to analyze based on input parameters.

        Selects indices either manually or by element type as specified in the parameter dictionary.

        Args:
            param (dict): Dictionary specifying selection method and parameters.

        Returns:
            list or np.array: List of atom indices to analyze.
        """

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
        """
        Return manually specified atom indices.

        Args:
            param (dict): Dictionary containing 'idx_list'.

        Returns:
            list: List of atom indices.
        """
        idx_list = param.get("idx_list")
        return idx_list

    def get_idx_list_all(self, param):
        """
        Return all atom indices of a specified element.

        Args:
            param (dict): Dictionary containing 'element'.

        Returns:
            np.array: Array of atom indices for the specified element.
        """
        element = param.get("element")
        idx_list = self.u.select_atoms(f'type {element}')
        return idx_list

    def get_atom_density(self, param, idx_list):
        """
        Calculate and save the atom density profile.

        Computes the histogram of atom z-coordinates, normalizes the density,
        and saves the profile to a file.

        Args:
            param (dict): Parameters for density calculation (e.g., bin width, unit).
            idx_list (list or np.array): Indices of atoms to include in the density profile.
        """
        logger.info("START GETTING ATOM DENSITY")
        logger.info("----------------------------")
        # dz: default 0.05 resolution for density profile
        # dz is different for different systems.
        dz = param.get("dz", 0.05)
        atom_z = self.all_z.T[idx_list]
        # atoms are aligned to the left surface
        # TODO: we can also align to the center of water.
        atom_z = atom_z - self.surf1_z_list

        # A certain atom_z possibly exceeds cell boundary, need wrap its value.
        atom_z_new = []
        cell_z = self.cellpar[2]
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

        # throw the last one and shift half bin
        z = z[:-1] + dz/2

        density_unit = param.get("density_unit", "number")
        unit_conversion = self.get_unit_conversion(density_unit, dz, self.xy_area)


        # normalized wrt density of bulk water
        density = density/self.n_frames * unit_conversion

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

    @staticmethod
    def get_unit_conversion(density_unit, dz, xy_area):
        """
        Compute the unit conversion factor for density.

        Args:
            density_unit (str): Desired density unit ('water' or 'number').
            dz (float): Bin width along z-axis.
            xy_area (float): Area of the xy-plane.

        Returns:
            float: Conversion factor for density normalization.
        """
        if density_unit == "water":
            bulk_density = 32/9.86**3
            unit_conversion = xy_area*dz*bulk_density
            unit_conversion = 1.0/unit_conversion
        elif density_unit == "number":
            unit_conversion = 1.0

        return unit_conversion

    def get_ave_density(self, width_list):
        all_cent_density = {}
        all_cent_density[f"width"] = width_list
        for name, density in self.atom_density.items():
            z = self.atom_density_z[name]
            cent = z[-1]/2
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
        plt.style.use('cp2kdata.matplotlibstyle.jcp')
        cp2kdata_cb_lcmap = mpl.colormaps['cp2kdata_cb_lcmap']
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cp2kdata_cb_lcmap.colors)
        fig = plt.figure(figsize=(3.37, 1.89), dpi=400, facecolor='white')
        ax = fig.add_subplot()
        for name, density in self.atom_density.items():
            z = self.atom_density_z[name]
            density = density
            if sym:
                density = (np.flip(density) + density)/2
            ax.plot(z, density, label=name)
        ax.legend()
        ax.set_ylabel("Density")
        ax.set_xlabel(r"z ($\mathrm{\AA}$)")
        return fig


# old density analysis function/class

def analysis_run(args):
    if args.CONFIG:
        import json
        logger.info("Analysis Start!")
        with open(args.CONFIG, 'r') as f:
            inp = json.load(f)
        system = Density(inp)
        system.run()
        logger.info("FINISHED")


class Density():
    def __init__(self, inp):
        logger.info("Perform Atom Density Analysis")
        # print file name
        self.twoD = False

        # this is not refactored
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