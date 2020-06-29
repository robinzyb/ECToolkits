from ase.io import read, write
from ase.geometry.analysis import Analysis
import numpy as np
import os



def set_pbc(pos, cell):
    """set pbc for Atoms object"""
    for single_pos in pos:
        single_pos.set_cell(cell)
        single_pos.set_pbc(True)

def take_tot_rdf(pos, r, nbin, frames, elements):
    tmp_info = Analysis(pos)
    tmp_rdf_list = tmp_info.get_rdf(r, nbin, imageIdx=slice(0, frames, 1), elements=elements)
    tot_gr = np.zeros(nbin)
    for s_gr in tmp_rdf_list:
        tot_gr += s_gr/frames
    return tot_gr

def xyz2rdf(in_name, out_name, cell, r, nbin, elements, frames = None):
    pos = read(in_name, index= ":")
    print(len(pos))
    set_pbc(pos, cell)
    if frames == None:
        frames = len(pos)
    rdf = take_tot_rdf(pos, r, nbin, frames, elements)
    r_list = np.arange(0, r, r/nbin)
    np.savetxt(out_name, np.array([r_list, rdf]).T)


in_file_name = ["./phenol/pot1-chlabel.xyz", "./phenol/pot2-chlabel.xyz", "./phenol/pot3-chlabel.xyz", "./phenol/pot4-chlabel.xyz", "./phenol/pot5-chlabel.xyz"]
out_file_name = ["./phenol/gr_1_O-O.dat", "./phenol/gr_2_O-O.dat", "./phenol/gr_3_O-O.dat", "./phenol/gr_4_O-O.dat", "./phenol/gr_5_O-O.dat"]
cell = [14.57000, 14.57000, 14.57000, 90, 90, 90]
r = 7
nbin = 200
elements = ["O", "O"]

for i in range(5):
    print(in_file_name[i])
    xyz2rdf(in_file_name[i], out_file_name[i], cell, r, nbin, elements)
    print("write", out_file_name[i], "finished")
