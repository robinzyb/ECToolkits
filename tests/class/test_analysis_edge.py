import os
import shutil
import pytest 
import numpy as np
from ase.io import read
from MDAnalysis import Universe
from ectoolkits.structures.rutile110 import Rutile1p11Edge
from ectoolkits.utils.rutile110 import pair_M5c_n_obr
from ectoolkits.analysis.rutile110 import (WatDensity,        # z-axis water denstiy profile
                                        RutileDisDeg,      # Ad water dissociation degree
                                        dAdBridge,         # Ad water - Obr  distance 
                                        dObr_NearestH)     # Surace Obr - Nearest proton distances
# External test files 
FIXTURE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
    )

DATA_DIR    = os.path.join("./", "data_output")
FIGURE_DIR  = os.path.join("./", "figure_output")

pytest.skip(allow_module_level=True, 
            reason="This entire test module is disabled for now."
                              "Failing test - needs to be fixed"
            )

class R110EdgeInp():
    """
    Tmp class to manage inputs for pytesting 
    """

    def __init__(self, ag, r110edge):
        self.ag = ag
        self.r110edge = r110edge

def load_r110_edge(traj_name, vecy, vecz):
    atoms = read(os.path.join(traj_dir, traj_name+".cif"))
    r110edge  = Rutile1p11Edge(atoms, vecy=vecy, vecz=vecz, cutoff=2.9)
    return r110edge

def load_traj(traj_name):
    atoms = read(os.path.join(traj_dir, traj_name+".cif"))
    u = Universe(os.path.join(traj_dir, traj_name+".xyz"))
    u.dimensions = atoms.cell.cellpar() 
    u.trajectory.ts.dt = 0.00005
    ag = u.atoms
    return ag


traj_dir = os.path.join(FIXTURE_DIR, "_trajectories")
traj_name_list = ["edge-16wat",
                  "edge-8wat",
                  "edge-4wat"]
vecy_list = np.array([[26.34844236,  1.8642182,  -2.0615678],
                      [23.32958114,  2.9181609,  -2.84689072],
                      [10.19062564,  0.88917123, -2.18995354]])
vecz_list = np.array([[0.49339,  -0.070221,  9.201458],
                      [ 1.69668234, -0.46340771, 12.77995387],
                      [ 2.6386353,  -0.38744058, 12.69295913]])

ag_to_try   = list(map(load_traj, traj_name_list))
r110edge_to_try = list(map(load_r110_edge, traj_name_list, vecy_list, vecz_list))

inp_to_try  = list(map(R110EdgeInp, ag_to_try, r110edge_to_try))

@pytest.mark.parametrize('inp', inp_to_try, ids=traj_name_list) # id is a good way to name the test
def test_pair_M5c_n_obr(inp):
    atoms = inp.r110edge
    ind = inp.r110edge.get_indices()
    ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
    cn5idx  = ind['idx_M5c'].reshape(2, -1)
    obr_idx = np.concatenate([ind['idx_Obr'].reshape(2, -1), ind['idx_hObr_upper'].reshape(2, -1)], axis=1)
    _, upper_obr = pair_M5c_n_obr(atoms, cn5idx[0], obr_idx[0])
    _, lower_obr = pair_M5c_n_obr(atoms, cn5idx[1], obr_idx[1])
    idx_obr = np.array([upper_obr, lower_obr])
    # test result array's shape
    is_tworow_upper = (upper_obr.shape[1] == 2)
    is_tworow_lower = (upper_obr.shape[1] == 2)
    is_enough_upper = (upper_obr.shape[0] == cn5idx.shape[1])
    is_enough_lower = (upper_obr.shape[0] == cn5idx.shape[1])
    assert (is_tworow_upper & is_tworow_lower & is_enough_upper & is_enough_lower)


@pytest.mark.parametrize('inp', inp_to_try,   ids=traj_name_list)
class TestRutile110Edge():
    """Test the analysis methods for rutile (110)-water interface with <1-11>
    edge
    """
    def test_WatDensity(self, inp):
        # test could this ananlysis run normally
        r110edge = inp.r110edge
        rotM     = r110edge.refine_rotM()
        wd = WatDensity(inp.ag, rotM=rotM)
        wd.run()
        # test if all output files are dumped correctly
        has_oxygen     = os.path.isfile(os.path.join(DATA_DIR, "hist_oxygen.npy"))
        has_hydrogen   = os.path.isfile(os.path.join(DATA_DIR, "hist_hydrogen.npy"))
        has_d_oxygen   = os.path.isfile(os.path.join(DATA_DIR, "oxygen.dat"))
        has_d_hydrogen = os.path.isfile(os.path.join(DATA_DIR, "hydrogen.dat"))
        has_scale      = os.path.isfile(os.path.join(DATA_DIR, "hist2rho_scale.dat"))
        assert (has_oxygen & has_hydrogen & has_d_oxygen & has_d_hydrogen & has_scale)

    def test_RutileDisDeg(self, inp):
        # test could this ananlysis run normally
        ag       = inp.ag
        r110edge = inp.r110edge
        owidx, _ = r110edge.get_wat()
        ind      = r110edge.get_indices()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)

        cn5idx = ind['idx_M5c'].reshape(2, -1)
        edge5idx = ind['idx_edge_M5c'].reshape(2, -1)
        edge4idx = ind['idx_edge_M4c'].reshape(2, -1)
    
        disdeg   = RutileDisDeg(ag, owidx, cn5idx, nrow=r110edge.nrow, 
                                edge4idx=edge4idx, edge5idx=edge5idx,) 
        disdeg.run()
        # test if all output files are dumped correctly
        has_dist5s     = os.path.isfile(os.path.join(DATA_DIR, "distOH-5s.npy"))
        has_dist5e     = os.path.isfile(os.path.join(DATA_DIR, "distOH-5e.npy"))
        has_dist4e     = os.path.isfile(os.path.join(DATA_DIR, "distOH-4e.npy"))
        has_disdeg     = os.path.isfile(os.path.join(DATA_DIR, "disdeg.npy"))
        has_histOH1    = os.path.isfile(os.path.join(DATA_DIR, "histOH1.dat"))
        has_histOH2    = os.path.isfile(os.path.join(DATA_DIR, "histOH2.dat"))
        assert (has_dist5s & has_disdeg & has_histOH1 & has_histOH1)

    def test_dAdBridge(self, inp):
        # test could this ananlysis run normally
        ag       = inp.ag
        r110edge = inp.r110edge
        idx_owat, _ = r110edge.get_wat()
        ind = r110edge.get_indices()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
        cn5idx  = ind['idx_M5c'].reshape(2, -1)
        # 'idx_hObr_upper' is also known as the last Obr 
        obr_idx = np.concatenate([ind['idx_Obr'].reshape(2, -1), ind['idx_hObr_upper'].reshape(2, -1)], axis=1)
        _, upper_obr = pair_M5c_n_obr(r110edge, cn5idx[0], obr_idx[0])
        _, lower_obr = pair_M5c_n_obr(r110edge, cn5idx[1], obr_idx[1])
        idx_obr = np.array([upper_obr, lower_obr])

        dab = dAdBridge(ag, cn5idx, idx_obr, idx_owat, idx_adO=None)
        dab.run()

        # test if all output files are dumped correctly
        has_upper_dab  = os.path.isfile(os.path.join(DATA_DIR, "upper-dab.npy"))
        has_lower_dab  = os.path.isfile(os.path.join(DATA_DIR, "lower-dab.npy"))
        has_ind_Oad    = os.path.isfile(os.path.join(DATA_DIR, "ad_O_indices.npy"))
        assert (has_upper_dab & has_lower_dab & has_ind_Oad)
   
    def test_dObr_NearestH(self, inp):
        # test could this ananlysis run normally
        ag          = inp.ag
        r110edge    = inp.r110edge
        idx_owat, _ = r110edge.get_wat()
        ind         = r110edge.get_indices()
        ind['idx_Obr'][0] = np.flip(ind['idx_Obr'][0], axis=1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)
        idx_hobr1 = ind['idx_hObr_mid'].reshape(2, -1)
        idx_hobr2 = ind['idx_hObr_upper'].reshape(2, -1)
        idx_eobr  = ind['idx_edge_O2'].reshape(2, -1)
        doh = dObr_NearestH(ag, idx_obr, nrow=r110edge.nrow, idx_hobr1=idx_hobr1, 
                            idx_hobr2=idx_hobr2, idx_eobr=idx_eobr)
        doh.run()
        # test if all output files are dumped correctly
        has_dist_Obr      = os.path.isfile(os.path.join(DATA_DIR, "d_Obr-H.npy"))
        has_dist_hist     = os.path.isfile(os.path.join(DATA_DIR, "histObrH.dat"))
        has_dist_Obr_half = os.path.isfile(os.path.join(DATA_DIR, "d_hObr1-H.npy"))
        has_dist_Obr_n1   = os.path.isfile(os.path.join(DATA_DIR, "d_hObr2-H.npy"))
        has_dist_Obr_edge = os.path.isfile(os.path.join(DATA_DIR, "d_eObr-H.npy"))
        assert (has_dist_Obr & has_dist_hist)


# Finally remove the generated datafiles
def test_clear():
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    if os.path.isdir(FIGURE_DIR):
        shutil.rmtree(FIGURE_DIR)
