import json
from pathlib import Path

import pytest
import numpy as np
from ase.io import read
from MDAnalysis import Universe

from ectoolkits.analysis.proton_transfer_cv import ProtonTransferCV


path_prefix = Path("tests/analysis/proton_transfer_cv/")
system_list = ["bivo4-direct-pt",]

@pytest.fixture(params=system_list, ids=system_list, scope='class')
def analysis_and_reference(request):
    # tmp_path cannot be used here since it is function scope
    system = request.param
    system_dir = path_prefix/system
    with open(system_dir/"input.json", "r") as f:
        input_param = json.load(f)

    # modify the input to find the correct files
    traj_file = system_dir/input_param["traj_file"]
    # readlammps
    stc = read(traj_file, format="lammps-dump-text", specorder=["Bi", "H", "O", "V"])
    chemical_symbols = stc.get_chemical_symbols()
    u = Universe(traj_file, format="LAMMPSDUMP")
    u.add_TopologyAttr("types", np.array(chemical_symbols))
    atomgroup = u.atoms

    # read types
    idxs_type1_o = input_param["idxs_type1_o"]
    idxs_type2_o = input_param["idxs_type2_o"]
    analysis = ProtonTransferCV(atomgroup=atomgroup, idxs_type1_o=idxs_type1_o, idxs_type2_o=idxs_type2_o)
    analysis.run()

    # reference
    reference_file = system_dir/"cv_reference.npy"
    reference = np.load(reference_file)
    return analysis, reference

class TestProtonTransferCV():
    def test_proton_transfer_cv(self, analysis_and_reference):
        analysis = analysis_and_reference[0]
        ptcv_ref = analysis_and_reference[1]
        ptcv = analysis.results
        np.testing.assert_array_equal(ptcv, ptcv_ref)






