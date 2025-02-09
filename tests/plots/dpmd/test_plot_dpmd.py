from pathlib import Path
import filecmp

import pytest
import numpy as np

from ectoolkits.plots.dpmd import plot_dptest, plot_model_devi


path_prefix_dptest = Path("tests/plots/dpmd/dptest_files")
case_dptest_list = ["dptest"]
@pytest.fixture(params=case_dptest_list, ids=case_dptest_list, scope='class')
def case_dptest(request):
    return request.param


class TestPlotDPMD():
    def test_plot_dptest(self, case_dptest, tmp_path):
        e_file = path_prefix_dptest/f"{case_dptest}.e.out"
        f_file = path_prefix_dptest/f"{case_dptest}.f.out"
        save_png = tmp_path/f"{case_dptest}.png"
        fig, rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz = \
            plot_dptest(e_file=e_file, f_file=f_file, save_name=save_png, return_err=True, frc_comp=True)
        
        err = np.array([rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz])
        ref = np.load(path_prefix_dptest/f"{case_dptest}_err.npy")
        np.testing.assert_array_equal(err, ref)


path_prefix_model_devi = Path("tests/plots/dpmd/model_devi_files")
case_model_devi_list = ["model_devi"]
@pytest.fixture(params=case_model_devi_list, ids=case_model_devi_list, scope='class')
def case_model_devi(request):
    return request.param

class TestPlotModelDevi():

    
    def test_plot_model_devi(self, case_model_devi, tmp_path):
        model_devi_files = [
            [path_prefix_model_devi/"iter-000"/f"{case_model_devi}-0.out", path_prefix_model_devi/"iter-000"/f"{case_model_devi}-1.out"],
            [path_prefix_model_devi/"iter-001"/f"{case_model_devi}-0.out", path_prefix_model_devi/"iter-001"/f"{case_model_devi}-1.out"]
            ]
        save_png = tmp_path/f"{case_model_devi}.png"
        fig = plot_model_devi(model_devi_files=model_devi_files, trust_lo=0.2, trust_hi=0.8, save_name=save_png)

        assert save_png.exists()