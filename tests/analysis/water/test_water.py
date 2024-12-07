"""
Testing water orientation analysis 
"""

import json
import numpy as np
import pytest
from ase.cell import Cell
from ectoolkits.analysis.water import WaterOrientation

# Define the different test systems and their corresponding files
test_systems = [
    {
        "name": "P-water",
        "xyz-file": "tests/analysis/atom_density/P-water/P-water.xyz",
        "input-file": "tests/analysis/atom_density/P-water/input.json",
        "reference-file": "tests/analysis/water/P-water.dat",
    },
    {
        "name": "SnO2-water",
        "xyz-file": "tests/analysis/atom_density/SnO2-water/SnO2-water.xyz",
        "input-file": "tests/analysis/atom_density/SnO2-water/input.json",
        "reference-file": "tests/analysis/water/SnO2-water.dat",
    },
]


@pytest.fixture(
    params=test_systems,
    ids=[d["name"] for d in test_systems],
)
def name_analysis_answer(request):
    """
    Get water orientation class object
    """
    name = request.param["name"]
    xyz_file = request.param["xyz-file"]
    json_file = request.param["input-file"]
    reference_file = request.param["reference-file"]

    with open(json_file, "r", encoding="utf-8") as f:
        input_dict = json.load(f)
    return (
        name,
        WaterOrientation(
            xyz=xyz_file,
            cell=Cell.new(input_dict["cell"]),
            surf1=input_dict["surf1"],
            surf2=input_dict["surf2"],
            strict=True,
            dz=0.05,
        ),
        reference_file,
    )


def test_water_orientation_profiles(name_analysis_answer, tmp_path):
    """
    Pytest comparing water orientation outputs to the reference .dat files
    """
    # Get reference data for the current system
    name, analysis, reference_file = name_analysis_answer

    # Create a temporary file path for saving profiles
    output_file = tmp_path / "temp.dat"

    # Run the WaterOrientation analysis
    analysis.run()

    # Save the profiles to the temporary path
    analysis.save_profiles(output_file)

    # Load both profiles
    _, rho, costheta = np.loadtxt(output_file, unpack=True)
    _, rho_ref, costheta_ref = np.loadtxt(reference_file, unpack=True)

    # Use np.testing to compare arrays with tolerance
    np.testing.assert_allclose(
        rho,
        rho_ref,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Density profile mismatch for {name}",
    )

    np.testing.assert_allclose(
        costheta,
        costheta_ref,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Orientation profile mismatch for {name}",
    )
