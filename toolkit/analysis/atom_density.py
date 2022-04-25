import numpy as np
import matplotlib.pyplot as plt
import os

from ase.io import read, write
from ase.neighborlist import neighbor_list

from toolkit.utils import fancy_print
from toolkit.slab import Slab


def analysis_run(args):
    if args.CONFIG:
        import json
        fancy_print("Analysis Start!")
        with open(args.CONFIG, 'r') as f:
            inp = json.load(f)
        system = AtomDensity(inp)
        system.run()
        fancy_print("FINISHED")




class AtomDensity():
    def __init__(self, inp):
        fancy_print("Perform Atom Density Analysis")
        # print file name
        self.xyz_file = inp.get("xyz_file")
        if not os.path.isfile(self.xyz_file):
            raise FileNotFoundError

        fancy_print("Read Structure File: {0}".format(inp["xyz_file"]))
        # print the cell
        self.cell = inp["cell"]
        fancy_print("Read the Cell Info: {0}".format(inp["cell"]))
        # print surface 1
        self.surf1 = inp["surf1"]
        self.surf1 = np.array(self.surf1)
        fancy_print("Read Surface 1 Atoms Index: {0}".format(inp["surf1"]))
        # print surface 2
        self.surf2 = inp["surf2"]
        self.surf2 = np.array(self.surf2)
        fancy_print("Read Surface 2 Atoms Index: {0}".format(inp["surf2"]))

#        self.shift_center = inp["shift_center"]
#        fancy_print("Density will shift center to water center: {0}".format(self.shift_center))

        # slice for stucture
        index = inp.get("nframe", ":")

        self.density_type = inp.get("density_type")

        # Start reading structure
        fancy_print("Now Start Reading Structures")
        fancy_print("----------------------------")
        self.poses = read(self.xyz_file, index=index)
        self.poses[0] = Slab(self.poses[0])
        fancy_print("Reading Structures is Finished")

        self.nframe = len(self.poses)
        fancy_print("Read Frame Number: {0}".format(self.nframe))
        self.natom = len(self.poses[0])
        fancy_print("Read Atom Number: {0}".format(self.natom))

        # Check the Index method
        #self.atom_idx_method = inp["O_density"]["O_index_method"]
        # get the idx

        # if self.O_idx_method == "GEUSS_init":
        #     fancy_print("Use the Oxygen index of water guessed from coordination number")
        #     self.Ow_idx = self.water_O_idx()
        # elif self.O_idx_method == "external":
        #     fancy_print("Use the Oxygen index from external input")
        #     self.Ow_idx = self.external_idx(inp)



    def run(self):
        # read all structure and corresponding z
        self.all_z = self.get_all_z()

        # cell info
        self.cell_volume = self.poses[0].get_volume()
        self.xy_area = self.cell_volume/self.cell[2]

        # surface 1 and 2 position along the trajectory
        self.surf1_z_list = self.get_surf1_z_list()
        self.surf2_z_list = self.get_surf2_z_list()
        self.surf1_z = self.surf1_z_list.mean()
        self.surf2_z = self.surf2_z_list.mean()

        # find the water center along the trajectory
        self.water_cent_list = self.get_water_cent_list()
        

        # water center relative to fisrt frame
        #self.water_cent_rel_s = self.water_cent_s - self.water_cent_s[0]

        #fancy_print("Calculated Origin Water Center Position: {0} A".format(self.water_cent))
        #fancy_print("Water Center will shift to Cell Center: {0} A".format(self.cell[2]/2))

        for param in self.density_type:
            idx_list = self.get_idx_list(param)
            self.get_atom_density(param, idx_list=idx_list)


        #self.get_o_density()
        #self.dump_o_density()

    def get_all_z(self)->np.array :
        """get the z coordinates of atoms along trajectory

        _extended_summary_

        Returns:
            np.array: the z coordinates of atoms
        """        
        
        all_z = []
        for pos in self.poses:
            # wrap the cell
            pos.set_cell(self.cell)
            pos.wrap()
            #z
            all_z.append(pos.get_positions().T[2])
        all_z = np.stack(all_z)
        return all_z


    def get_surf1_z_list(self)->np.array:
        """calculate the surface 1 average position

        _extended_summary_

        Returns:
            np.array: axis 0: traj
        """        

        surf1_z_list = self.all_z.T[self.surf1]
        surf1_z_list = surf1_z_list.T
        surf1_z_list = surf1_z_list.mean(axis=1)
        return surf1_z_list

    def get_surf2_z_list(self)->np.array:
        """calculate the surface 2 average position

        _extended_summary_

        Returns:
            np.array: axis 0: traj
        """        

        surf2_z_list = self.all_z.T[self.surf2]
        surf2_z_list = surf2_z_list.T
        surf2_z_list = surf2_z_list.mean(axis=1)
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
        i,j = neighbor_list('ij', self.poses[0], {('O', 'H'): 1.3})
        cn = np.bincount(i)

        H2O_pair_list = []
        Ow_idx = np.where(cn == 2)[0]
        np.savetxt(os.path.join(os.path.dirname(self.xyz_file), "Ow_idx.dat"), Ow_idx, fmt='%d')
        return Ow_idx

    def get_idx_list(self, param):
        # get the idx from external input

        idx_method = param.get("idx_method")
        if idx_method == "manual":
            idx_list = self.get_idx_list_manual(param)
        elif idx_method == "all":
            idx_list = self.get_idx_list_all(param)
        else:
            fancy_print("Not implement")
            raise ValueError
        return idx_list

    def get_idx_list_manual(self, param):
        idx_list = param.get("idx_list")
        return idx_list

    def get_idx_list_all(self, param):
        element = param.get("element")
        idx_list = self.poses[0].find_element_idx_list(element=element)
        return idx_list

    # @property
    # def z_shift(self):
    #     return self.water_cent - self.cell[2]/2

    #def wrap_number(self, num):
    #     if num < 0:
    #         num += self.cell[2]
    #     elif num > self.cell[2]:
    #         num -= self.cell[2]
    #     return num

    def get_atom_density(self, param, idx_list):
        fancy_print("START GETTING ATOM DENSITY")
        fancy_print("----------------------------")

        dz = param.get("dz", 0.05)

        atom_z = self.all_z.T[idx_list]
#        atom_z = atom_z.T
        # shift the atom_z to eliminate the effect of slab drift
        # atom_z = atom_z - self.water_cent_rel_s
        atom_z = atom_z - self.surf1_z_list

        # this might cause the num exceed cell boundary, need wrap the number
        atom_z_new = []
        cell_z = self.cell[2]
        for num in atom_z.flatten():
            atom_z_new.append(num%cell_z)
        atom_z = np.array(atom_z_new)

        
        # find the length between two surface
        if self.surf1_z > self.surf2_z:
            self.surf_space = self.surf2_z + self.cell[2] - self.surf1_z
        else:
            self.surf_space = self.surf2_z - self.surf1_z

        bins = int(self.surf_space/dz)

        density, z = np.histogram(atom_z, bins=bins, range=(0, self.surf_space))

        # throw the last one and move the number half bin
        z = z[:-1] + dz/2
        #z = z[:-1]


        
        unit_conversion = self.get_unit_conversion(param, dz)

        # normalized wrt density of bulk water
        density = density/self.nframe * unit_conversion

        
        # shift the center to water center
        #if self.shift_center:
        #    z = z - self.z_shift
        #    shift_idx = np.where(z<0)
        #    z[shift_idx] = z[shift_idx] + self.cell[2]

        #    z = np.roll(z, -len(shift_idx[0]))
        #    density = np.roll(density, -len(shift_idx[0]))


        self.atom_density = density
        self.atom_density_z = z
        element = param.get("element")
        output_file = param.get("output", f"{element}_output.dat")
        np.savetxt(
                output_file,
                np.stack((self.atom_density_z, self.atom_density)).T,
                header="FIELD: z[A], atom_density"
        )
        fancy_print(f"Density Profile Data Save to {output_file}")

    def get_unit_conversion(self, param, dz):
        density_unit = param.get("density_unit", "number")
        if density_unit == "water":
            bulk_density = 32/9.86**3
            unit_conversion = self.xy_area*dz*bulk_density
            unit_conversion = 1.0/unit_conversion
        elif density_unit == "number":
            unit_conversion = 1.0

        return unit_conversion


    def dump_o_density(self):
        fancy_print("---------------------------")
        fancy_print("START PLOT ATOM DENSITY")
        plt.figure()
        plt.plot(self.atom_density_z, self.atom_density)
        plt.show()
        #plt.savefig(os.path.join(os.path.dirname(self.xyz_file), "o_density.pdf"))
        #fancy_print("Oxygen Density Profile Save to o_density.pdf")

