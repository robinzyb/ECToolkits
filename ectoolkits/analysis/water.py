"""
Water analysis
"""

import numpy as np

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis import Universe
from MDAnalysis import transformations as trans

from ase.geometry import get_distances
from ase.cell import Cell

from ectoolkits.structures.slab import Slab


def get_dipoles(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    cell: Cell,
    oh_cutoff: float,
) -> np.ndarray:
    """
    TODO: To write
    """
    oh_vectors, oh_distances = get_distances(
        h_positions, o_positions, cell=cell, pbc=True
    )

    # For each H atom, find the indices of the closest O atom
    closest_o_indices = np.argmin(oh_distances, axis=1)
    min_oh_vectors = oh_vectors[np.arange(oh_vectors.shape[0]), closest_o_indices]
    min_oh_distance = oh_distances[np.arange(oh_vectors.shape[0]), closest_o_indices]

    # Find the indices of oxygen atoms corresponding to 2 H's (water) within cutoff
    unique, counts = np.unique(
        closest_o_indices[min_oh_distance < oh_cutoff],
        return_counts=True,
    )
    water_oxygen_indices = unique[counts == 2]

    # Boolean array to select H atoms that are in water
    h_mask = np.isin(closest_o_indices, water_oxygen_indices)

    # Select only O-indices corresponding to water
    selected_closest_o_indices = closest_o_indices[h_mask]

    # Select only OH vectors corresponding to water molecules
    water_oh_vectors = min_oh_vectors[h_mask]

    # Calculate dipole vector as the sum of two OH vectors for each water-O
    dipole_vectors = np.zeros(o_positions.shape[0])

    for i in water_oxygen_indices:
        dipole_vectors[i, :] = np.sum(
            water_oh_vectors[selected_closest_o_indices == i], axis=0
        )

    return dipole_vectors


class WaterOrientation(AnalysisBase):
    """
    TODO: Write
    """

    def __init__(
        self,
        topology: Slab,
        xyz: str,
        verbose: bool = False,
        **kwargs,
    ):
        self.universe = Universe(
            xyz,
            transformations=trans.boxdimensions.set_dimensions(topology.cell.cellpar()),
        )

        super().__init__(self.universe.trajectory, verbose=verbose)
        self.n_frames = len(self.universe.trajectory)
