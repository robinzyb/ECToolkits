"""
Water analysis
"""

import numpy as np
from scipy import constants

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis import Universe
from MDAnalysis import transformations as trans

from ase.geometry import get_distances
from ase.cell import Cell

from ectoolkits.log import get_logger

logger = get_logger(__name__)


def density(n, v, mol_mass: float):
    """
    calculate density (g/cm^3) from the number of particles

    Parameters
    ----------
    n : int or array
        number of particles
    v : float or array
        volume
    mol_mass : float
        mole mass in g/mol
    """
    rho = (n / constants.Avogadro * mol_mass) / (
        v * (constants.angstrom / constants.centi) ** 3
    )
    return rho


def water_density(n, v):
    """
    TODO: write
    """
    return density(n, v, 18.015)


def bin_edges_to_grid(bin_edges: np.ndarray):
    """
    TODO: write
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def get_dipoles(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    cell: Cell,
    oh_cutoff: float,
) -> np.ndarray:
    """
    TODO: To write
    """
    # Precompute all OH vectors
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
    dipole_vectors = np.ones(o_positions.shape) * np.nan

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
        xyz: str,
        cell: Cell,
        surf1: np.ndarray,
        surf2: np.ndarray,
        **kwargs,
    ):
        logger.info("Class for water density and orientation analysis")

        # Setup Universe
        universe = Universe(
            xyz,
            transformations=trans.boxdimensions.set_dimensions(cell.cellpar()),
        )

        # Save required arguments
        self.cell = cell
        self.surf1_ag = universe.select_atoms(*[f"index {i}" for i in surf1])
        self.surf2_ag = universe.select_atoms(*[f"index {i}" for i in surf2])
        logger.info("Surface 1 atom indices: %s", str(surf1))
        logger.info("Surface 2 atom indices: %s", str(surf2))

        # Initialize AnalysisBase
        super().__init__(universe.trajectory, verbose=kwargs.get("verbose", False))
        self.n_frames = len(universe.trajectory)
        logger.info("Number of frames: %d", self.n_frames)

        # Parse optional kwargs
        self.oh_cutoff = kwargs.get("oh_cutoff", 2)
        # self.reference = kwargs.get("reference", "fixed")
        # logger.info("Coordinate reference method: %s", self.reference)

        # Define atom groups
        o_indices = kwargs.get("oxygen_indices", None)
        if o_indices:
            self.o_ag = universe.select_atoms(*[f"index {i}" for i in o_indices])
        else:
            self.o_ag = universe.select_atoms("name O")

        h_indices = kwargs.get("hydrogen_indices", None)
        if h_indices:
            self.h_ag = universe.select_atoms(*[f"index {i}" for i in h_indices])
        else:
            self.h_ag = universe.select_atoms("name H")

        # Initialize results
        self.results.z_water = np.zeros((self.n_frames, len(self.o_ag)))
        self.results.cos_theta = np.zeros((self.n_frames, len(self.o_ag)))
        self.results.dipole = np.zeros((self.n_frames, len(self.o_ag), 3))
        self.results.z_surf = np.zeros((self.n_frames, 2))

    def _single_frame(self):
        """
        Compute surface position, water density and cos theta for a single frame
        """
        # Surface position
        z_surf = [
            np.mean(self.surf1_ag.positions[:, 2]),
            np.mean(self.surf2_ag.positions[:, 2]),
        ]
        np.copyto(self.results.z_surf[self._frame_index, :], z_surf)

        # Oxygen density
        np.copyto(self.results.z_water[self._frame_index], self.o_ag.positions[:, 2])

        # Dipoles
        dipole = get_dipoles(
            self.h_ag.positions,
            self.o_ag.positions,
            cell=self.cell,
            oh_cutoff=self.oh_cutoff,
        )
        np.copyto(self.results.dipole[self._frame_index], dipole)

        # Cos theta
        cos_theta = (dipole[:, 2]) / np.linalg.norm(dipole, axis=-1)
        np.copyto(self.results.cos_theta[self._frame_index], cos_theta)

    def _conclude(self):
        # Surface area
        area = self.cell.area(2)

        # Surface locations
        z_surf = self.results.z_surf.mean(axis=0)
        z1 = min(z_surf)
        z2 = max(z_surf)

        # Water density
        counts, bin_edges = np.histogram(
            self.results.z_water.flatten(),
            bins=int((z2 - z1) / 0.1),
            range=(z1, z2),
        )
        n_water = counts / self.n_frames
        grid_volume = np.diff(bin_edges) * area
        rho = water_density(n_water, grid_volume)
        self.results.density_profile = [bin_edges_to_grid(bin_edges), rho]

        # Water orientation
        valid = ~np.isnan(self.results.cos_theta.flatten())
        counts, bin_edges = np.histogram(
            self.results.z_water.flatten()[valid],
            bins=int((z2 - z1) / 0.1),
            range=(z1, z2),
            weights=self.results.cos_theta.flatten()[valid],
        )
        self.results.orientation_profile = [
            bin_edges_to_grid(bin_edges),
            counts / self.n_frames,
        ]
