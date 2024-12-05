"""
Water analysis
"""

from typing import Dict, List

import numpy as np
from scipy import constants

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis import Universe
from MDAnalysis import transformations as trans
from MDAnalysis.lib.distances import minimize_vectors, capped_distance

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


def identify_water_molecules(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    cell: Cell,
    oh_cutoff: float,
) -> Dict[int, List]:
    """
    TODO: write
    """
    water_dict = {i: [] for i in range(o_positions.shape[0])}

    # TODO: get rid of this for-loop
    for h_idx, hpos in enumerate(h_positions):
        pairs, distances = capped_distance(
            hpos,
            o_positions,
            max_cutoff=oh_cutoff,
            box=cell.cellpar(),
            return_distances=True,
        )

        if len(pairs) > 0:
            closest_o_idx = pairs[np.argmin(distances)][1]
            water_dict[closest_o_idx].append(h_idx)

    water_dict = {key: value for key, value in water_dict.items() if len(value) == 2}
    return water_dict


def get_dipoles(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    water_dict: Dict[int, List],
    cell: Cell,
) -> np.ndarray:
    """
    TODO: write
    """
    o_indices = np.array([key for key, _ in water_dict.items()])
    h_indices = np.array([value for _, value in water_dict.items()])

    oh_1 = minimize_vectors(
        o_positions[o_indices] - h_positions[h_indices[:, 0]],
        box=cell.cellpar(),
    )
    oh_2 = minimize_vectors(
        o_positions[o_indices] - h_positions[h_indices[:, 1]],
        box=cell.cellpar(),
    )
    dipoles = np.ones(o_positions.shape) * np.nan
    dipoles[o_indices, :] = oh_1 + oh_2
    return dipoles


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
        logger.info("Performing water density and orientation analysis")

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
        super().__init__(universe.trajectory, verbose=kwargs.get("verbose", None))
        self.n_frames = len(universe.trajectory)

        # Parse optional kwargs
        self.oh_cutoff = kwargs.get("oh_cutoff", 2)
        # self.reference = kwargs.get("reference", "fixed")
        # logger.info("Coordinate reference method: %s", self.reference)
        # BUG: the surfaces might drift, requiring different reference coordinate

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

        self.water_dict = identify_water_molecules(
            self.h_ag.positions,
            self.o_ag.positions,
            self.cell,
            oh_cutoff=self.oh_cutoff,
        )
        self.strict = kwargs.get("strict", False)

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
        if self.strict:
            self.water_dict = identify_water_molecules(
                self.h_ag.positions,
                self.o_ag.positions,
                self.cell,
                oh_cutoff=self.oh_cutoff,
            )
        dipole = get_dipoles(
            self.h_ag.positions,
            self.o_ag.positions,
            self.water_dict,
            cell=self.cell,
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
        # BUG: if run(step=N), some elements are zero, leading to wrong z1/z2
        z1 = min(z_surf)
        z2 = max(z_surf)

        # Water density
        counts, bin_edges = np.histogram(
            self.results.z_water.flatten(),
            bins=int((z2 - z1) / 0.1),
            range=(z1, z2),
        )
        n_water = counts / self.n_frames
        # BUG: if run(step=N), some elements are zero, leading to wrong counts/n_frames ratio
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
