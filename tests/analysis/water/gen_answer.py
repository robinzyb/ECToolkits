"""
Generate answers for water density test
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.cell import Cell
from ectoolkits.analysis.water import WaterOrientation

parent = Path("../atom_density")
system_list = ["P-water", "SnO2-water"]

fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

for i, system in enumerate(system_list):
    system_dir = parent / Path(system)
    with open(system_dir / "input.json", "r", encoding="utf-8") as f:
        input_data = json.load(f)

    wo = WaterOrientation(
        xyz=system_dir / f"{system}.xyz",
        cell=Cell.new(input_data["cell"]),
        surf1=input_data["surf1"],
        surf2=input_data["surf2"],
        strict=True,
        dz=0.05,
    )
    wo.run()
    wo.save_profiles(f"{system}.dat")

    # For the oxide, the water density profile differs from the oxygen atom profile
    # because some oxygen atoms are (de)protonated. A comparison is plotted below.

    wo.plot_density(fig.axes[i])
    ad = np.loadtxt(system_dir / "water_density.dat")
    fig.axes[i].plot(ad[:, 0], ad[:, 1], "k", linewidth=1, label="AtomDensity")
    fig.axes[i].set_title(system)

fig.tight_layout()
plt.show()
