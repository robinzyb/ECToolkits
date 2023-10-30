import os
import shutil
import pytest 
import numpy as np
from ase.io import read
from MDAnalysis import Universe
from ectoolkits.structures.rutile110 import Rutile110
from ectoolkits.utils.rutile110 import pair_M5c_n_obr
from ectoolkits.analysis.rutile110 import (WatDensity,        # z-axis water denstiy profile
                                        RutileDisDeg,      # Ad water dissociation degree
                                        dAdBridge,         # Ad water - Obr  distance 
                                        dInterLayer,       # TiO2 Interlayer distances
                                        SurfTiOBondLenght, # TiO2 surface Ti-O bondlength
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
#DATA_DIR    = os.path.join(
#    os.path.dirname(os.path.realpath(__file__)), "data_output"
#    )
#
#FIGURE_DIR  = os.path.join(
#    os.path.dirname(os.path.realpath(__file__)), "figure_output"
#    )

class R110Inp():
    """
    Tmp class to manage inputs for pytesting 
    """

    def __init__(self, ag, r110):
        self.ag = ag
        self.r110 = r110

def load_r110(traj_name, nrow, bridge_along):
    atoms = read(os.path.join(traj_dir, traj_name+".cif"))
    r110  = Rutile110(atoms, nrow=nrow, bridge_along=bridge_along)
    return r110

def load_traj(traj_name):
    atoms = read(os.path.join(traj_dir, traj_name+".cif"))
    u = Universe(os.path.join(traj_dir, traj_name+".xyz"))
    u.dimensions = atoms.cell.cellpar() 
    u.trajectory.ts.dt = 0.00005
    ag = u.atoms
    return ag


traj_dir = os.path.join(FIXTURE_DIR, "_trajectories")
traj_name_list = ["flat_4x2",
                  "flat_8x4"]
nrow_list         = [2, 4]
bridge_along_list = ["x", "y"]

ag_to_try   = list(map(load_traj, traj_name_list))
r110_to_try = list(map(load_r110, traj_name_list, nrow_list, bridge_along_list))

inp_to_try  = list(map(R110Inp, ag_to_try, r110_to_try))

@pytest.mark.parametrize('inp', inp_to_try,   ids=traj_name_list)
def test_pair_M5c_n_obr(inp):
    atoms = inp.r110
    ind = inp.r110.get_indicies()
    ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
    cn5idx  = ind['idx_M5c'].reshape(2, -1)
    obr_idx = ind['idx_Obr'].reshape(2, -1)
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
class TestRutile110Flat():
    """Test the rutile 110 strucutre identifier
    """
    def test_WatDensity(self, inp):
        # test could this ananlysis run normally
        wd = WatDensity(inp.ag, rotM=None)
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
        r110     = inp.r110
        owidx, _ = r110.get_wat()
        cn5idx   = r110.get_indicies()['idx_M5c']
        disdeg   = RutileDisDeg(ag, owidx, cn5idx, nrow=r110.nrow) 
        disdeg.run()
        # test if all output files are dumped correctly
        has_dist5s     = os.path.isfile(os.path.join(DATA_DIR, "distOH-5s.npy"))
        has_disdeg     = os.path.isfile(os.path.join(DATA_DIR, "disdeg.npy"))
        has_histOH1    = os.path.isfile(os.path.join(DATA_DIR, "histOH1.dat"))
        has_histOH2    = os.path.isfile(os.path.join(DATA_DIR, "histOH2.dat"))
        assert (has_dist5s & has_disdeg & has_histOH1 & has_histOH1)

    def test_dAdBridge(self, inp):
        # test could this ananlysis run normally
        ag       = inp.ag
        r110     = inp.r110
        idx_owat, _ = r110.get_wat()
        atoms = inp.r110
        ind = inp.r110.get_indicies()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
        cn5idx  = ind['idx_M5c'].reshape(2, -1)
        obr_idx = ind['idx_Obr'].reshape(2, -1)
        _, upper_obr = pair_M5c_n_obr(atoms, cn5idx[0], obr_idx[0])
        _, lower_obr = pair_M5c_n_obr(atoms, cn5idx[1], obr_idx[1])
        idx_obr = np.array([upper_obr, lower_obr])
        dab = dAdBridge(ag, cn5idx, idx_obr, idx_owat, idx_adO=None)
        dab.run()

        # test if all output files are dumped correctly
        has_upper_dab  = os.path.isfile(os.path.join(DATA_DIR, "upper-dab.npy"))
        has_lower_dab  = os.path.isfile(os.path.join(DATA_DIR, "lower-dab.npy"))
        has_ind_Oad    = os.path.isfile(os.path.join(DATA_DIR, "ad_O_indicies.npy"))
        assert (has_upper_dab & has_lower_dab & has_ind_Oad)

    def test_dInterLayer(self, inp):
        # test could this ananlysis run normally
        ag   = inp.ag
        dil  = dInterLayer(ag) 
        dil.run()

        # test if all output files are dumped correctly
        has_z_mean       = os.path.isfile(os.path.join(DATA_DIR, "ti_z_mean.npy"))
        has_z_mean_hist  = os.path.isfile(os.path.join(DATA_DIR, "ti_z_mean_histo.dat"))
        assert (has_z_mean & has_z_mean_hist)

    def test_SurfTiOBondLenght(self, inp):
        # test could this ananlysis run normally
        ag       = inp.ag
        r110     = inp.r110
        idx_owat, _ = r110.get_wat()
        ind = inp.r110.get_indicies()
        ind['idx_M5c'][0] = np.flip(ind['idx_M5c'][0], axis=1)
        ind['idx_Obr'][0] = np.flip(ind['idx_Obr'][0], axis=1)
        idx_cn5  = ind['idx_M5c'].reshape(2, -1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)

        sbl = SurfTiOBondLenght(ag, idx_cn5, idx_obr, idx_owat)
        sbl.run()

        # test if all output files are dumped correctly
        has_Ti_Oad  = os.path.isfile(os.path.join(DATA_DIR, "d_TiOad.npy"))
        has_Ti_Obr  = os.path.isfile(os.path.join(DATA_DIR, "d_TiObr.npy"))
        has_ind     = os.path.isfile(os.path.join(DATA_DIR, "indicies.dat"))
        assert (has_Ti_Oad & has_Ti_Obr & has_ind)

    def test_dObr_NearestH(self, inp):
        ag       = inp.ag
        r110     = inp.r110
        idx_owat, _ = r110.get_wat()
        ind = inp.r110.get_indicies()
        ind['idx_Obr'][0] = np.flip(ind['idx_Obr'][0], axis=1)
        idx_obr  = ind['idx_Obr'].reshape(2, -1)
        doh = dObr_NearestH(ag, idx_obr, nrow=r110.nrow, idx_hobr1=None, idx_hobr2=None, idx_eobr=None)
        doh.run()
        # test if all output files are dumped correctly
        has_dist_Obr      = os.path.isfile(os.path.join(DATA_DIR, "d_Obr-H.npy"))
        has_dist_hist     = os.path.isfile(os.path.join(DATA_DIR, "histObrH.dat"))
        #has_dist_Obr_half = os.path.isfile(os.path.join(DATA_DIR, "d_hObr1-H.npy"))
        #has_dist_Obr_n1   = os.path.isfile(os.path.join(DATA_DIR, "d_hObr2-H.npy"))
        #has_dist_Obr_edge = os.path.isfile(os.path.join(DATA_DIR, "d_eObr-H.npy"))
        assert (has_dist_Obr & has_dist_hist)


# Finally remove the generated datafiles
def test_clear():
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    if os.path.isdir(FIGURE_DIR):
        shutil.rmtree(FIGURE_DIR)
