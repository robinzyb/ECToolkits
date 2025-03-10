import json
from pathlib import Path

import numpy as np
from ase.io import read
from MDAnalysis import Universe

from ectoolkits.analysis.proton_transfer_cv import ProtonTransferCV

system_list = ["bivo4-direct-pt",]

for system in system_list:
    system_dir = Path(system)
    input_file = system_dir/"input.json"
    with open(input_file, "r") as f:
        input_param = json.load(f)

    traj_file = system_dir/input_param["traj_file"]
    stc = read(traj_file, format="lammps-dump-text", specorder=["Bi", "H", "O", "V"])
    chemical_symbols = stc.get_chemical_symbols()
    u = Universe(traj_file, format="LAMMPSDUMP")
    u.add_TopologyAttr("types", np.array(chemical_symbols))
    atomgroup = u.atoms

    idxs_type1_o = input_param["idxs_type1_o"]
    idxs_type2_o = input_param["idxs_type2_o"]
    analysis = ProtonTransferCV(atomgroup=atomgroup, idxs_type1_o=idxs_type1_o, idxs_type2_o=idxs_type2_o)
    analysis.run()

    save_file = system_dir/"cv_reference.npy"
    np.save(save_file, analysis.results)

