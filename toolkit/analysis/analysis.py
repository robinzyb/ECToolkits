import numpy as np
import matplotlib.pyplot as plt
import os

from ase.io import read, write

from toolkit.utils import fancy_print


def analysis_run(args):
    if args.CONFIG:
        fancy_print("Analysis Start!")
        system = Analysis(args.CONFIG)
        system.run()
        fancy_print("FINISHED")


class Analysis():
    def __init__(self, inp_file):

        import json
        with open(inp_file, 'r') as f:
            inp = json.load(f)


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

        self.shift_center =inp["shift_center"]
        fancy_print("Density will shift center to water center: {0}".format(self.shift_center))

        # how many structures will be read
        if "nframe" in inp:
            index = '::{0}'.format(inp["nframe"])
        else:
            index = ':'
        # Start reading structure
        self.poses = read(inp["xyz_file"], index=index)

        self.nframe = len(self.poses)
        fancy_print("Read Frame Number: {0}".format(self.nframe))
        self.natom = len(self.poses[0])
        fancy_print("Read Atom Number: {0}".format(self.natom))

    def run(self):
        # read all structure and corresponding z
        self.all_z = []
        for pos in self.poses:
            pos.set_cell(self.cell)
            pos.wrap()
            #z
            self.all_z.append(pos.get_positions().T[2])

        # cell info
        self.cell_volume = self.poses[0].get_volume()
        self.xy_area = self.cell_volume/self.cell[2]

        # turn into np array
        self.all_z = np.stack(self.all_z)
        if self.surf1_ave > self.surf2_ave:
            self.water_cent = (self.surf2_ave + self.cell[2] + self.surf1_ave)/2
        else:
            self.water_cent = (self.surf2_ave + self.surf1_ave)/2
        fancy_print("Calculated Origin Water Center Position: {0} A".format(self.water_cent))
        fancy_print("Water Center will shift to Cell Center: {0} A".format(self.cell[2]/2))

        if self.shift_center:
            self.surf1_ave_shift = self.wrap_number(self.surf1_ave - self.z_shift)
            self.surf2_ave_shift = self.wrap_number(self.surf2_ave - self.z_shift)
        else:
            # no shift one, just for convenience
            self.surf1_ave_shift = self.surf1_ave
            self.surf2_ave_shift = self.surf2_ave
        self.get_o_density()
        self.dump_o_density()




    @property
    def surf1_ave(self):
        # calculate the surface 1 average position

        surf1_z = self.all_z.T[self.surf1-1]
        surf_ave = surf1_z.flatten().mean()
        return surf_ave

    @property
    def surf2_ave(self):
        # calculate the surface 2 average position

        surf2_z = self.all_z.T[self.surf2-1]
        surf_ave = surf2_z.flatten().mean()
        return surf_ave

    @property
    def o_idx(self):
        o_idx = []
        for at in self.poses[0]:
            if at.symbol == 'O':
                o_idx.append(at.index)
        return o_idx

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
        fancy_print("---------------------------")
        fancy_print("START GETING OXYGEN DENSITY")

        o_z = self.all_z.T[self.o_idx]
        o_z = o_z.T
        dz = 0.2
        # get the bin number
        bins = int(self.cell[2]/dz)

        density, z = np.histogram(o_z.flatten(), bins=bins)

        # throw the last one and move the number half bin
        z = z[:-1] + dz/2
        #z = z[:-1]

        # the density of bulk water
        bulk_density = 32/9.86**3

        # normalized wrt density of bulk water
        density = density/(self.xy_area*dz)/self.nframe/bulk_density

        # shift the center to water center
        if self.shift_center:
            z = z - self.z_shift
            shift_idx = np.where(z<0)
            z[shift_idx] = z[shift_idx] + self.cell[2]

            z = np.roll(z, -len(shift_idx[0]))
            density = np.roll(density, -len(shift_idx[0]))


        self.o_density = density
        self.o_density_z = z

    def dump_o_density(self):
        fancy_print("---------------------------")
        fancy_print("START PLOT OXYGEN DENSITY")
        plt.figure()
        plt.plot(self.o_density_z, self.o_density)
        plt.vlines(self.surf1_ave_shift , 0, 10)
        plt.vlines(self.surf2_ave_shift , 0, 10)
        plt.savefig("o_density.pdf")
        fancy_print("Oxygen Density Profile Save to o_density.pdf")

