from pathlib import Path
import filecmp

import pytest
import numpy as np

from ectoolkits.plots.dpmd import plot_dptest



path_prefix = Path("tests/plots/dpmd")
case_list = ["dptest"]

@pytest.fixture(params=case_list, ids=case_list, scope='class')
def case(request):
    return request.param

class TestPlotDPMD():
    def test_plot_dptest(self, case, tmp_path):
        e_file = path_prefix/f"{case}.e.out"
        f_file = path_prefix/f"{case}.f.out"
        save_png = tmp_path/f"{case}.png"
        fig, rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz = \
            plot_dptest(e_file=e_file, f_file=f_file, save_name=save_png, return_err=True, frc_comp=True)
        
        err = np.array([rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz])
        ref = np.load(path_prefix/f"{case}_err.npy")
        np.testing.assert_array_equal(err, ref)
