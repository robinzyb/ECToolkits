import os
import shutil
from typing import Tuple, List

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.io import read, write
from ase.build import surface
from MDAnalysis.lib.distances import minimize_vectors

from ectoolkits.utils.math import get_plane_eq
from ectoolkits.log import get_logger

logger = get_logger(__name__)


class Slab(Atoms):
    """
        Object inherent from Atoms object in ASE.
        Add method for vacuum slab model
    Args:
        Atoms (_type_): Atoms object int ASE
    """

    def get_cus(self, input_idx, coord_num, cutoff):
        """
        function to get atom index of coordinate unsaturated sites.
        slab: Atoms object, the slab model
        input_idx: the index of the atom you want get the coordination number
        coord_num: coordination number for coordinate unsaturated sites, the number must be less then the full coordination
        cutoff: the cutoff radius defining coordination. something like: {('Ti', 'O'): 2.2}
        return: the index for cus atoms
        """
        coord_num_list = np.bincount(
            neighbor_list('i', self, cutoff))[input_idx]
        target_idx = input_idx[coord_num_list == coord_num]
        return target_idx

    def get_neighbor_list(self, idx: int, cutoff: dict) -> list:
        """
            provided that atom index and return its neighbor in list
        Args:
            idx (int): atom index
            cutoff (dict): cutoff for neighbor pair

        Returns:
            list: list of atom indices

        Examples:
            get_neighbor_list(16, {("O", "H"): 1.4})
        """
        i, j = neighbor_list('ij', self, cutoff=cutoff)
        return j[i == idx]

    def find_element_idx_list(self, element: str) -> list:
        """
        find atom index provided that element symbol

        _extended_summary_

        Args:
            element (str): element symbol

        Returns:
            list: list of atom indices
        """
        cs = self.get_chemical_symbols()
        cs = np.array(cs)
        idx_list = np.where(cs == element)[0]
        return list(idx_list)

    def find_surf_idx(self,
                      element: str = None,
                      tolerance: float = 0.1,
                      dsur: str = 'up',
                      check_cross_boundary=False,
                      trans_z_dist=5
                      ) -> list:
        """
            find atom indexs at surface

        _extended_summary_

        Args:
            element (str): element symbol
            tolerance (float, optional): tolerance for define a layer. Defaults to 0.1.
            dsur (str, optional): direction of surface, 'up' or 'dw'. for a vacuum-slab model,
            you have up surface and down surface. Defaults to 'up'.

        Returns:
            list: list of atom indices
        """
        tmp_stc = self.copy()
        if check_cross_boundary:
            while tmp_stc.is_cross_z_boundary(element=element):
                logger.info(
                    f"The slab part is cross z boundary, tranlate {trans_z_dist:3.3f} A!")
                tmp_stc.translate([0, 0, trans_z_dist])
                tmp_stc.wrap()

        if element:
            idx_list = tmp_stc.find_element_idx_list(element)
            z_list = tmp_stc[idx_list].get_positions().T[2]
        else:
            z_list = tmp_stc.get_positions().T[2]
        if dsur == 'up':
            z = z_list.max()
        elif dsur == 'dw':
            z = z_list.min()

        zmin = z-tolerance
        zmax = z+tolerance
        idx_list = tmp_stc.find_idx_from_range(
            zmin=zmin, zmax=zmax, element=element)

        return idx_list

    def del_surf_layer(self, element: str = None, tolerance=0.1, dsur='up', check_cross_boundary=False):
        """ delete the layer atoms,

        _extended_summary_

        Args:
            element (str, optional): _description_. Defaults to None.
            tolerance (float, optional): _description_. Defaults to 0.1.
            dsur (str, optional): _description_. Defaults to 'up'.

        Returns:
            _type_: _description_
        """

        del_list = self.find_surf_idx(element=element,
                                      tolerance=tolerance,
                                      dsur=dsur,
                                      check_cross_boundary=check_cross_boundary
                                      )

        tmp = self.copy()
        del tmp[del_list]
        return tmp

    def find_idx_from_range(self, zmin: int, zmax: int, element: str = None) -> list:
        """_summary_

        _extended_summary_

        Args:
            zmin (int): minimum in z
            zmax (int): maximum in z
            element (str, optional): element symbol, None means all atoms. Defaults to None.

        Returns:
            list: list of atom indices
        """
        idx_list = []
        if element:
            for atom in self:
                if atom.symbol == element:
                    if (atom.position[2] < zmax) and (atom.position[2] > zmin):
                        idx_list.append(atom.index)
        else:
            for atom in self:
                if (atom.position[2] < zmax) and (atom.position[2] > zmin):
                    idx_list.append(atom.index)
        return idx_list

    def del_from_range(self, zmin: int, zmax: int, element: str = None) -> Atoms:
        """_summary_

        _extended_summary_

        Args:
            zmin (int): _description_
            zmax (int): _description_
            element (str, optional): _description_. Defaults to None.

        Returns:
            Atoms: _description_
        """
        tmp = self.copy()
        del_idx_list = self.find_idx_from_range(
            zmin=zmin, zmax=zmax, element=element)

        del tmp[del_idx_list]

        return tmp

    def add_adsorbate(self,
                      ad_site_idx: int,
                      vertical_dist: float,
                      adsorbate: Atoms,
                      contact_atom_idx: int = 0,
                      lateral_shift: Tuple[float] = (0, 0),
                      ):

        tmp_stc = self.copy()
        site_pos = tmp_stc[ad_site_idx].position.copy()
        tmp_adsorbate = adsorbate.copy()
        # refer the positions of adsorbate to the contact_atom
        contact_atom_pos = tmp_adsorbate[contact_atom_idx].position.copy()
        tmp_adsorbate.translate(-contact_atom_pos)
        # move the adsorbate to target position
        target_pos = site_pos + \
            np.array([lateral_shift[0], lateral_shift[1], vertical_dist])
        tmp_adsorbate.translate(target_pos)
        tmp_stc.extend(tmp_adsorbate)
        return tmp_stc

    def add_adsorbates(self,
                       ad_site_idx_list: List[int],
                       vertical_dist: float,
                       adsorbate: Atoms,
                       contact_atom_idx: int = 0,
                       lateral_shift: Tuple[float] = (0, 0),
                       ):
        tmp_stc = self.copy()
        for ad_site_idx in ad_site_idx_list:
            tmp_stc = tmp_stc.add_adsorbate(ad_site_idx=ad_site_idx,
                                            vertical_dist=vertical_dist,
                                            adsorbate=adsorbate,
                                            contact_atom_idx=contact_atom_idx,
                                            lateral_shift=lateral_shift,
                                            )
        return tmp_stc

    def generate_interface(self,
                           water_box_len: float,
                           top_surface_idx: List[int],
                           bottom_surface_idx: List[int]
                           ):
        """merge slab model and water box together

        Args:
            water_box_len:
            top_surface_idx:
            bottom_surface_idx:

        Returns:
            tmp:
        """
        # find the water box
        if os.path.exists("gen_water/watbox.xyz"):
            water_box = read("gen_water/watbox.xyz")
            logger.info("Water Box Found")
        else:
            logger.info("Water Box Not Found")
            raise FileNotFoundError(
                'Water box not found, please install packmol')

        tmp = self.copy()
        cell_z = tmp.get_cell()[2][2]
        # shift the water in z directions (to add in slab model)
        tmp_water_positions = water_box.get_positions()
        for i in range(len(tmp_water_positions)):
            tmp_water_positions[i] += [0, 0, cell_z + 0.5]
        water_box.set_positions(tmp_water_positions)
        # add the water box to slab model
        tmp.extend(water_box)
        # modify the z length
        tmp.set_cell(tmp.get_cell() +
                     [[0, 0, 0], [0, 0, 0], [0, 0, water_box_len + 1]])
        # shift the water center to box center
        top_surface_z = tmp[top_surface_idx].get_positions().T[2].mean()
        bottom_surface_z = tmp[bottom_surface_idx].get_positions().T[2].mean()
        slab_center_z = 0.5 * (top_surface_z + bottom_surface_z)
        tmp.translate([0, 0, -slab_center_z])
        tmp.set_pbc([False, False, True])
        tmp.wrap()
        logger.info("Merge Water and Slab Box Finished")
        return tmp

    def generate_water_box(self, water_box_len):
        """function to generate water box
        x and y length is from self length
        Args:
            water_box_len:

        Returns:

        """
        cell = self.get_cell()
        cell_a = cell[0]
        cell_b = cell[1]
        header = "-"
        logger.info(header * 50)
        logger.info("Now Generate Water Box")
        space_per_water = 9.86 ** 3 / 32
        wat_num = (np.linalg.norm(np.cross(cell_a, cell_b))
                   * water_box_len) / space_per_water
        wat_num = int(wat_num)
        # logger.info("Read Cell X: {0:03f} A".format(cell_a))
        # logger.info("Read Cell Y: {0:03f} A".format(cell_b))
        logger.info("Read Water Box Length: {0:03f} A".format(water_box_len))
        logger.info("Predict Water Number: {0}".format(wat_num))

        n_vec_a, d1_a, d2_a, n_vec_b, d1_b, d2_b = get_plane_eq(cell_a, cell_b)
        logger.info("Calculate Plane Equation")

        if os.path.exists('gen_water'):
            logger.info("found gen_water direcotry, now remove it")
            shutil.rmtree('gen_water')
        logger.info("Generate New Directory: gen_water")
        os.mkdir('gen_water')
        logger.info("Generate Packmol Input: gen_wat_box.inp")
        with open(os.path.join("gen_water", "gen_wat_box.inp"), 'w') as f:
            txt = "#packmol input generate by python"
            txt += "\n"
            txt += "tolerance 2.0\n"
            txt += "filetype xyz\n"
            txt += "output watbox.xyz"
            txt += "\n"
            txt += "structure water.xyz\n"
            txt += "  number {0}\n".format(int(wat_num))
            txt += "  above plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(
                n_vec_a[0], n_vec_a[1], n_vec_a[2], d1_a+0.5)
            txt += "  below plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(
                n_vec_a[0], n_vec_a[1], n_vec_a[2], d2_a-0.5)
            txt += "  above plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(
                n_vec_b[0], n_vec_b[1], n_vec_b[2], d1_b+0.5)
            txt += "  below plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(
                n_vec_b[0], n_vec_b[1], n_vec_b[2], d2_b-0.5)
            txt += "  above plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(
                0, 0, 1.0, 0.)
            txt += "  below plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(
                0, 0, 1.0, water_box_len)
            txt += "end structure\n"
            f.write(txt)
        logger.info("Generate A Water Molecule: water.xyz")
        with open(os.path.join("gen_water", "water.xyz"), 'w') as f:
            txt = '3\n'
            txt += ' water\n'
            txt += ' H            9.625597       6.787278      12.673000\n'
            txt += ' H            9.625597       8.420323      12.673000\n'
            txt += ' O           10.203012       7.603800      12.673000\n'
            f.write(txt)
        logger.info("Generate Water Box: watbox.xyz")
        os.chdir("./gen_water")
        os.system("packmol < gen_wat_box.inp")
        os.chdir("../")
        logger.info("Generate Water Box Finished")

    def remove_cell_vacuum(self, adopt_space=2):
        """remove the vacuum of z direction
         cell z must be perpendicular to xy plane
        """
        tmp = self.copy()
        z_list = tmp.get_positions().T[2]
        slab_length = z_list.max() - z_list.min()
        slab_length += 2
        a = tmp.get_cell()[0]
        b = tmp.get_cell()[1]
        c = [0, 0, slab_length]
        tmp.set_cell([a, b, c])
        tmp.center()
        return tmp

    def is_cross_z_boundary(
        self,
        element: str = None
    ):
        # check if slab cross z boundary
        if element:
            M_idx_list = self.find_element_idx_list(element=element)
        else:
            M_idx_list = list(range(len(self)))

        cellpar = self.cell.cellpar()

        coords = self[M_idx_list].get_positions()
        coords_z = coords[:, 2]

        coord_z_max = coords[coords_z.argmax()]
        coord_z_min = coords[coords_z.argmin()]
        vec_raw = coord_z_max - coord_z_min

        vec_minimized = minimize_vectors(vectors=vec_raw, box=cellpar)

        # logger.info(vec_minimized[2], vec_raw[2])
        if np.isclose(vec_minimized[2], vec_raw[2], atol=1e-5, rtol=0):
            return False
        else:
            return True


class RutileSlab(Slab):
    """
    class atoms used for rutile like(structure) system
    space group: P42/mnm
    Usage:
    rutile = read("Rutile-exp.cif")
    x = RutileType(rutile)
    slab = []
    for i in range(3, 7):
        slab.append(x.get_slab(indices=(1, 1, 0), n_layers=i, lateral_repeat=(2, 4)))
    """

    def get_slab(self, indices: Tuple[int], n_layers, lateral_repeat: Tuple[int] = (2, 4), vacuum=10.0):
        h, k, l = indices
        entry = str(h)+str(k)+str(l)
        method_entry = {
            "110": self.rutile_slab_110,
            "001": self.rutile_slab_001,
            "100": self.rutile_slab_100,
            "101": self.rutile_slab_101,
        }

        method = method_entry.get(entry, None)

        try:
            assert method is not None
            slab = method(n_layers=n_layers,
                          lateral_repeat=lateral_repeat, vacuum=vacuum)
        except:
            logger.info("Current Miller Index has not implemented yet")
        # if method is None:
        #     raise ValueError("Current Miller Index has not implemented yet")

        return slab

    def rutile_slab_110(self, n_layers=5, lateral_repeat: tuple = (2, 4), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 110 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell

        slab = surface(self, (1, 1, 0), n_layers+1, vacuum)

        # remove bottom layer
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='up')

        # create the super cell
        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)

        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]

        return slab

    def rutile_slab_001(self, n_layers=5, lateral_repeat: tuple = (2, 2), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 001 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell

        if n_layers % 2 == 1:
            slab = surface(self, (0, 0, 1), int(n_layers/2)+1, vacuum)
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        elif n_layers % 2 == 0:
            slab = surface(self, (0, 0, 1), int(n_layers/2), vacuum)

        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)

        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]

        return slab

    def rutile_slab_100(self, n_layers=5, lateral_repeat: tuple = (2, 3), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 100 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell

        if n_layers % 2 == 1:
            slab = surface(self, (1, 0, 0), int(n_layers/2)+1, vacuum)
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
        elif n_layers % 2 == 0:
            slab = surface(self, (1, 0, 0), int(n_layers/2)+1, vacuum)
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)

        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]

        return slab

    def rutile_slab_101(self, n_layers=5, lateral_repeat: tuple = (2, 2), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 101 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell
        slab = surface(self, (1, 0, 1), n_layers+1, vacuum)
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='up')

        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)
        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]

        return slab
