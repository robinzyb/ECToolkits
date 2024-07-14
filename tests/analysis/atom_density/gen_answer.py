import json
from pathlib import Path

from ectoolkits.analysis.atom_density import AtomDensity

system_list = ["sno2-water"]

for system in system_list:
    system_dir = Path(system)
    input_file = system_dir/"input.json"
    with open(input_file, "r") as f:
        input_data = json.load(f)

    input_data["xyz_file"] = str(system_dir/f"{system}.xyz")
    input_data["density_type"][0]["name"] = str(system_dir/"water_density")
    ad = AtomDensity(input_data)
    ad.run()
