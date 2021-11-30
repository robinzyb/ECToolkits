import numpy as np
import matplotlib.pyplot as plt
import os

from ase.io import read, write
from ase.neighborlist import neighbor_list

from toolkit.utils import fancy_print


def analysis_run(args):
    if args.CONFIG:
        import json
        fancy_print("Analysis Start!")
        with open(args.CONFIG, 'r') as f:
            inp = json.load(f)
        system = Analysis(inp)
        system.run()
        fancy_print("FINISHED")


class Analysis():
    def __init__(self, inp):

        # print the function we choose
        self.functions = inp["functions"]
        fancy_print("The following analysis will be performed")
        for i in self.functions:
            fancy_print("Function: {0}".format(i))
        # print file name
        self.xyz_file = inp["xyz_file"]
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
        self.output = inp["output"]

#        self.shift_center = inp["shift_center"]
#        fancy_print("Density will shift center to water center: {0}".format(self.shift_center))

        # how many structures will be read
        if "nframe" in inp:
            index = '::{0}'.format(inp["nframe"])
        else:
            index = ':'

        # Start reading structure
        fancy_print("Now Start Reading Structures")
        fancy_print("----------------------------")
        self.poses = read(inp["xyz_file"], format='xyz', index=index)
        fancy_print("Reading Structures is Finished")

        self.nframe = len(self.poses)
        fancy_print("Read Frame Number: {0}".format(self.nframe))
        self.natom = len(self.poses[0])
        fancy_print("Read Atom Number: {0}".format(self.natom))

        # Check the Index method
        self.O_idx_method = inp["O_density"]["O_index_method"]
        # get the idx

        if self.O_idx_method == "GEUSS_init":
            fancy_print("Use the Oxygen index of water guessed from coordination number")
            self.Ow_idx = self.water_O_idx()
        elif self.O_idx_method == "external":
            fancy_print("Use the Oxygen index from external input")
            self.Ow_idx = self.external_idx(inp)
        else:
            fancy_print("Not implement")


    def run(self):
        # read all structure and corresponding z
        self.all_z = []
        for pos in self.poses:
            # wrap the cell
            pos.set_cell(self.cell)
            pos.wrap()
            #z
            self.all_z.append(pos.get_positions().T[2])

        # cell info
        self.cell_volume = self.poses[0].get_volume()
        self.xy_area = self.cell_volume/self.cell[2]

        # turn into np array
        self.all_z = np.stack(self.all_z)

        # surface 1 and 2 position along the trajectory
        self.surf1_ave_s = self.surf1_ave()
        self.surf2_ave_s = self.surf2_ave()

        self.surf1_ave = self.surf1_ave_s.mean()
        self.surf2_ave = self.surf2_ave_s.mean()

        # find the water center along the trajectory
        if self.surf1_ave > self.surf2_ave:
            self.water_cent_s = (self.surf2_ave_s + self.cell[2] + self.surf1_ave_s)/2
            self.water_cent = (self.surf2_ave + self.cell[2] + self.surf1_ave)/2
        else:
            self.water_cent_s = (self.surf2_ave_s + self.surf1_ave_s)/2
            self.water_cent = (self.surf2_ave + self.surf1_ave)/2

        # water center relative to fisrt frame
        self.water_cent_rel_s = self.water_cent_s - self.water_cent_s[0]

        fancy_print("Calculated Origin Water Center Position: {0} A".format(self.water_cent))
        fancy_print("Water Center will shift to Cell Center: {0} A".format(self.cell[2]/2))

#        if self.shift_center:
#            self.surf1_ave_shift = self.wrap_number(self.surf1_ave - self.z_shift)
#            self.surf2_ave_shift = self.wrap_number(self.surf2_ave - self.z_shift)
#        else:
#            # no shift one, just for convenience
#            self.surf1_ave_shift = self.surf1_ave
#            self.surf2_ave_shift = self.surf2_ave
        self.get_o_density()
        self.dump_o_density()




    def surf1_ave(self):
        # calculate the surface 1 average position

        surf1_z = self.all_z.T[self.surf1]
        surf_ave = surf1_z.mean(axis=0)
        return surf_ave

    def surf2_ave(self):
        # calculate the surface 2 average position

        surf2_z = self.all_z.T[self.surf2]
        surf_ave = surf2_z.mean(axis=0)
        return surf_ave

    def water_O_idx(self):
        # guess the o index of water
        i = neighbor_list('i', self.poses[0], {('O', 'H'): 1.3})
        j = neighbor_list('j', self.poses[0], {('O', 'H'): 1.3})
        cn = np.bincount(i)

        H2O_pair_list = []
        Ow_idx = np.where(cn == 2)[0]
        np.savetxt(os.path.join(os.path.dirname(self.xyz_file), "Ow_idx.dat"), Ow_idx, fmt='%d')
        return Ow_idx

    def external_idx(self, inp):
        # get the idx from external input

        O_idx = inp["O_density"]["O_index"]
        np.savetxt(os.path.join(os.path.dirname(self.xyz_file), "O_idx.dat"), O_idx, fmt='%d')
        return O_idx



    @property
    def z_shift(self):
        return self.water_cent - self.cell[2]/2

    def wrap_number(self, num):
        if num < 0:
            num += self.cell[2]
        elif num > self.cell[2]:
            num -= self.cell[2]
        return num

    def get_o_density(self):
        fancy_print("START GETTING OXYGEN DENSITY")
        fancy_print("----------------------------")


        o_z = self.all_z.T[self.Ow_idx]
#        o_z = o_z.T
        # shift the o_z to eliminate the effect of slab drift
        # o_z = o_z - self.water_cent_rel_s
        o_z = o_z - self.surf1_ave_s

        # this might cause the num exceed cell boundary, need wrap the number
        o_z_new = []
        for num in o_z.flatten():
            o_z_new.append(self.wrap_number(num))
        o_z = np.array(o_z_new)



        dz = 0.05
        # get the bin number

        # find the length between two surface
        if self.surf1_ave_s[0] > self.surf2_ave_s[0]:
            self.surf_space = self.surf2_ave + self.cell[2] - self.surf1_ave
        else:
            self.surf_space = self.surf2_ave - self.surf1_ave

        bins = int(self.surf_space/dz)

        density, z = np.histogram(o_z, bins=bins, range=(0, self.surf_space))

        # throw the last one and move the number half bin
        z = z[:-1] + dz/2
        #z = z[:-1]

        # the density of bulk water
        bulk_density = 32/9.86**3

        # normalized wrt density of bulk water
        density = density/(self.xy_area*dz)/self.nframe/bulk_density

        # shift the center to water center
        #if self.shift_center:
        #    z = z - self.z_shift
        #    shift_idx = np.where(z<0)
        #    z[shift_idx] = z[shift_idx] + self.cell[2]

        #    z = np.roll(z, -len(shift_idx[0]))
        #    density = np.roll(density, -len(shift_idx[0]))


        self.o_density = density
        self.o_density_z = z
        np.savetxt(
                os.path.join(os.path.dirname(self.xyz_file), self.output),
                np.stack((self.o_density_z, self.o_density)).T,
                header="FIELD: z[A], o_density"
        )
        fancy_print("Oxygen Density Profile Data Save to o_density.dat")

    def dump_o_density(self):
        fancy_print("---------------------------")
        fancy_print("START PLOT OXYGEN DENSITY")
        plt.figure()
        plt.plot(self.o_density_z, self.o_density)
        plt.savefig(os.path.join(os.path.dirname(self.xyz_file), "o_density.pdf"))
        fancy_print("Oxygen Density Profile Save to o_density.pdf")

