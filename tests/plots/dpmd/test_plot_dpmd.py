from pathlib import Path
import filecmp

import pytest

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
        plot_dptest(e_file=e_file, f_file=f_file, save_name=save_png)
        
        # here I just simply compre the two files.
        assert filecmp.cmp(save_png, 
                           path_prefix/f"{case}_ref.png",
                           )