from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np


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
    
