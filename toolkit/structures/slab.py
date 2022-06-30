from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.io import read, write
import numpy as np
import os
import shutil


class Slab(Atoms):
    """
        Object inherent from Atoms object in ASE.
        Add method for vacuum slab model
    Args:
        Atoms (_type_): Atoms object int ASE
    """

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
        return j[i==idx]

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
        idx_list = np.where(cs==element)[0]
        return list(idx_list)

    def find_surf_idx(self, element: str, tolerance=0.1, dsur='up') -> list:
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
        
        idx_list = self.find_element_idx_list(element)
        z_list = self[idx_list].get_positions().T[2]

        if dsur == 'up':
            z = z_list.max()
        elif dsur == 'dw':
            z = z_list.min()

        zmin = z-tolerance
        zmax = z+tolerance
        idx_list = self.find_idx_from_range(zmin=zmin, zmax=zmax, element=element)
    
        return idx_list

    def find_idx_from_range(self, zmin:int, zmax:int, element: str =None) -> list:
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

    def del_from_range(self, zmin: int, zmax: int, element: str =None) -> Atoms:
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
        del_idx_list = self.find_idx_from_range(zmin=zmin, zmax=zmax, element=element)

        del tmp[del_idx_list]
        
        return tmp
    
    def generate_interface(self, water_box_len, top_surface_idx, bottom_surface_idx):
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
            print("Water Box Found")
        else:
            print("Water Box Not Found")
            raise FileNotFoundError('Water box not found, please install packmol')

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
        tmp.set_cell(tmp.get_cell() + [[0, 0, 0], [0, 0, 0], [0, 0, water_box_len + 1]])
        # shift the water center to box center
        top_surface_z = tmp[top_surface_idx].get_positions().T[2].mean()
        bottom_surface_z = tmp[bottom_surface_idx].get_positions().T[2].mean()
        slab_center_z = 0.5 * (top_surface_z + bottom_surface_z)
        tmp.translate([0, 0, -slab_center_z])
        tmp.set_pbc([False, False, True])
        tmp.wrap()
        print("Merge Water and Slab Box Finished")
        return tmp
    
    def generate_water_box(self, water_box_len):
        """function to generate water box
        x and y length is from self length
        Args:
            water_box_len:

        Returns:

        """
        cell_x = self.get_cell()[0][0]
        cell_y = self.get_cell()[1][1]
        header = "-"
        print(header * 50)
        print("Now Generate Water Box")
        space_per_water = 9.86 ** 3 / 32
        wat_num = (cell_x * cell_y * water_box_len) / space_per_water
        wat_num = int(wat_num)
        print("Read Cell X: {0:03f} A".format(cell_x))
        print("Read Cell Y: {0:03f} A".format(cell_y))
        print("Read Water Box Length: {0:03f} A".format(water_box_len))
        print("Predict Water Number: {0}".format(wat_num))

        if os.path.exists('gen_water'):
            print("found gen_water direcotry, now remove it")
            shutil.rmtree('gen_water')
        print("Generate New Directory: gen_water")
        os.mkdir('gen_water')
        print("Generate Packmol Input: gen_wat_box.inp")
        with open(os.path.join("gen_water", "gen_wat_box.inp"), 'w') as f:
            txt = "#packmol input generate by python"
            txt += "\n"
            txt += "tolerance 2.0\n"
            txt += "filetype xyz\n"
            txt += "output watbox.xyz"
            txt += "\n"
            txt += "structure water.xyz\n"
            txt += "  number {0}\n".format(int(wat_num))
            txt += "  inside box 0. 0. 0. {0} {1} {2}\n".format(cell_x - 1, cell_y - 1, water_box_len)
            txt += "end structure\n"
            f.write(txt)
        print("Generate A Water Molecule: water.xyz")
        with open(os.path.join("gen_water", "water.xyz"), 'w') as f:
            txt = '3\n'
            txt += ' water\n'
            txt += ' H            9.625597       6.787278      12.673000\n'
            txt += ' H            9.625597       8.420323      12.673000\n'
            txt += ' O           10.203012       7.603800      12.673000\n'
            f.write(txt)
        print("Generate Water Box: watbox.xyz")
        os.chdir("./gen_water")
        os.system("packmol < gen_wat_box.inp")
        os.chdir("../")
        print("Generate Water Box Finished")

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