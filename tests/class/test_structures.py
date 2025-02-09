import os
import pytest 
import numpy as np
from ase.io import read

from ectoolkits.structures.rutile110 import (Rutile110, 
                                          Rutile1p11Edge)

# External test files 
FIXTURE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
    )

pytest.skip(allow_module_level=True, 
            reason="This entire test module is disabled for now."
                              "Failing test - needs to be fixed"
            )

# test flat Rutile110
flat_file_list = [
    os.path.join(FIXTURE_DIR, "_structures/4x2-flat-alongy.cif"),
    os.path.join(FIXTURE_DIR, "_structures/8x4-flat-alongy.cif"),
    os.path.join(FIXTURE_DIR, "_structures/4x2-flat-alongx.cif"),
]
flat_atoms_list    = list(map(read, flat_file_list))
rutile110_to_try   = (Rutile110(flat_atoms_list[0], nrow=2),
                      Rutile110(flat_atoms_list[1], nrow=4),
                      Rutile110(flat_atoms_list[2], nrow=2, bridge_along="x"))
rutile110_task_ids = ("4x2 supercell", "8x4 supercell", "4x2 supercell - obr along x-axis")

@pytest.mark.parametrize('r110', rutile110_to_try, ids=rutile110_task_ids)
class TestRutile110():
    """Test the rutile 110 strucutre identifier
    """
    def test_ind_keys(self, r110):
        ind = r110.get_indices()
        assert list(ind.keys()) == ['idx_M5c', 'idx_Obr']
    
    def test_shape(self, r110):
        ind    = r110.get_indices()
        shape1 = (ind['idx_M5c'].shape[:2] == (2, r110.nrow))
        shape2 = (ind['idx_Obr'].shape[:2] == (2, r110.nrow))
        assert (shape1 & shape2)

# test step Rutile1p11Edge
step_file_list = [
    os.path.join(FIXTURE_DIR, "_structures/16wat-edge.cif"),
    os.path.join(FIXTURE_DIR, "_structures/8wat-edge.cif"),
    os.path.join(FIXTURE_DIR, "_structures/4wat-edge.cif")
]
step_atoms_list = list(map(read, step_file_list))
rutile1p11_to_try = (Rutile1p11Edge(step_atoms_list[0], 
                                    vecy=np.array([26.34844236,  1.8642182,  -2.0615678]), 
                                    vecz=np.array([0.49339,  -0.070221,  9.201458])),
                     Rutile1p11Edge(step_atoms_list[1], 
                                    vecy=np.array([23.32958114,  2.9181609,  -2.84689072]), 
                                    vecz=np.array([ 1.69668234, -0.46340771, 12.77995387])), 
                     Rutile1p11Edge(step_atoms_list[2], 
                                    vecy=np.array([10.19062564,  0.88917123, -2.18995354]), 
                                    vecz=np.array([ 2.6386353,  -0.38744058, 12.69295913])))

rutile110_task_ids = ("16 water step", "8 water step", "4 water step")

@pytest.mark.parametrize('r110_step', rutile1p11_to_try, ids=rutile110_task_ids)
class TestRutile1p11Edge():
    """Test the rutile 110 strucutre identifier
    """
    def test_ind_keys(self, r110_step):
        ind = r110_step.get_indices()
        assert list(ind.keys()) == ['idx_M5c', 
                                    'idx_edge_M5c', 
                                    'idx_edge_M4c', 
                                    'idx_Obr', 
                                    'idx_hObr_mid', 
                                    'idx_hObr_upper', 
                                    'idx_edge_O2']
    
    def test_shape(self, r110_step):
        ind            = r110_step.get_indices()
        shape_Ti5s     = (ind['idx_M5c'].shape[:2] == (2, r110_step.nrow))
        shape_Obr      = (ind['idx_Obr'].shape[:2] == (2, r110_step.nrow))
        shape_Ti5e     = (ind['idx_edge_M5c'].shape   == (2, r110_step.nrow, 1))
        shape_Ti4e     = (ind['idx_edge_M4c'].shape   == (2, r110_step.nrow, 1))
        shape_Obr_last = (ind['idx_hObr_upper'].shape == (2, r110_step.nrow, 1))
        shape_Obr_half = (ind['idx_hObr_mid'].shape   == (2, r110_step.nrow, 1))
        shape_O2_edge  = (ind['idx_edge_O2'].shape    == (2, r110_step.nrow, 1))
        assert (shape_Ti5s & shape_Obr & shape_Ti5e & shape_Ti4e & shape_Obr_last & shape_Obr_half & shape_O2_edge)
