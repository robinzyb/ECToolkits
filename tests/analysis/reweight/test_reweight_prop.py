import json
from pathlib import Path

import pytest
import numpy as np

from ectoolkits.analysis.reweight import calc_property_surface

path_prefix = Path("tests/analysis/reweight/")
case_list = ["01.case_2d_property"]

@pytest.fixture(params=case_list, ids=case_list, scope='class')
def analysis_and_answer(request, tmp_path_factory):
    # tmp_path cannot be used here since it is function scope
    case = request.param
    case_dir = path_prefix/case
    colvar_file = case_dir/"COLVAR"
    prop_data_file = case_dir/"property_data.npy"
    prop_ref_file = case_dir/"prop_ref.dat"
    return colvar_file, prop_data_file, prop_ref_file


#TODO: I need to clearly define the input for the analysis. especially the unit of bias and kbT.
# save to file has a precision issue almost equal to 1e-6 is ok
class TestReweightFES():
    def test_fes(self, analysis_and_answer):
        colvar_file = analysis_and_answer[0]
        prop_data_file = analysis_and_answer[1]
        prop_ref_file = analysis_and_answer[2]

        cv1, cv2, bias = np.loadtxt(colvar_file, usecols=(1, 2, 3), unpack=True)
        prop_data = np.load(prop_data_file)
        grid_bin_x = 10
        grid_bin_y = 10
        T = 350
        kbT = T*0.0083144621 # kJ/mol
        cv1_min = 2.3
        cv1_max = 2.4
        cv2_min = 1.0
        cv2_max = 2.0
        bw_cv1 = 0.008
        bw_cv2 = 0.008

        prop = calc_property_surface(cv_x=cv1,
                                    cv_y=cv2,
                                    bias=bias,
                                    prop=prop_data, 
                                    bandwidth_x=bw_cv1, 
                                    bandwidth_y=bw_cv2,
                                    nbins_x=grid_bin_x, 
                                    nbins_y=grid_bin_y, 
                                    kbT=kbT, 
                                    save_file=False,
                                    min_cv_x=cv1_min, 
                                    max_cv_x=cv1_max, 
                                    min_cv_y=cv2_min, 
                                    max_cv_y=cv2_max,
                                    dim=2)    

        
        prop_ref = np.loadtxt(prop_ref_file, usecols=(2), unpack=True)
        np.testing.assert_array_almost_equal(prop.flatten(), prop_ref, decimal=6)