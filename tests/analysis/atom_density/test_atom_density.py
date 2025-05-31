import json
from pathlib import Path

import pytest
import numpy as np

from ectoolkits.analysis.atom_density import AtomDensity
from ectoolkits.analysis.atom_density import run_atom_density_analysis


path_prefix = Path("tests/analysis/atom_density/")
system_list = ["SnO2-water", "P-water"]

@pytest.fixture(params=system_list, ids=system_list, scope='class')
def analysis_and_answer(request, tmp_path_factory):
    # tmp_path cannot be used here since it is function scope
    system = request.param
    system_dir = path_prefix/system
    with open(system_dir/"input.json", "r") as f:
        input_param = json.load(f)

    # modify the input to find the correct files
    input_param["xyz_file"] = str(system_dir/f"{system}.xyz")
    water_density_file = str(tmp_path_factory.mktemp("data")/"water_density")
    input_param["density_type"][0]["name"] = water_density_file

    water_density_file_ref = system_dir/"water_density.dat"
  #  analysis = AtomDensity(input_param)
    return input_param, water_density_file_ref, water_density_file

class TestAtomDensity():
    def test_water_density(self, analysis_and_answer):
        input_param = analysis_and_answer[0]
        water_density_file_ref = analysis_and_answer[1]
        water_density_file = analysis_and_answer[2]
        run_atom_density_analysis(input_param)
        z, water_density = np.loadtxt(f"{water_density_file}.dat", unpack=True)
        z_ref, water_density_ref = np.loadtxt(water_density_file_ref, unpack=True)
        np.testing.assert_almost_equal(z, z_ref, decimal=6)
        np.testing.assert_almost_equal(water_density, water_density_ref, decimal=6)





