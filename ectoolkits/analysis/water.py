"""
Water analysis
"""

from typing import Dict, List

import numpy as np
from scipy import constants
from matplotlib.axes import Axes

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis import Universe
from MDAnalysis import transformations as trans
from MDAnalysis.lib.distances import minimize_vectors, capped_distance

from ase.cell import Cell

from ectoolkits.log import get_logger

logger = get_logger(__name__)


def density(n, v, mol_mass: float):
    """
    Calculate density in g/cm³ from the number of particles.

    Parameters
    ----------
    n : int or array-like
        Number of particles.
    v : float or array-like
        Volume in Å³.
    mol_mass : float
        Molar mass in g/mol.

    Returns
    -------
    float or array-like
        Density in g/cm³.
    """
    rho = (n / constants.Avogadro * mol_mass) / (
        v * (constants.angstrom / constants.centi) ** 3
    )
    return rho


def water_density(n, v):
    """
    Calculate the density of water in g/cm³.

    Parameters
    ----------
    n : int or array-like
        Number of water molecules.
    v : float or array-like
        Volume in Å³.

    Returns
    -------
    float or array-like
        Density in g/cm³.
    """
    return density(n, v, 18.015)


def bin_edges_to_grid(bin_edges: np.ndarray):
    """
    Convert bin edges to grid points at bin centers.

    Parameters
    ----------
    bin_edges : np.ndarray
        Array of bin edges.

    Returns
    -------
    np.ndarray
        Array of grid points at bin centers.
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def identify_water_molecules(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    cell: Cell,
    oh_cutoff: float,
) -> Dict[int, List[int]]:
    """
    Identify water molecules based on proximity of hydrogen and oxygen atoms.

    Parameters
    ----------
    h_positions : np.ndarray
        Positions of hydrogen atoms.
    o_positions : np.ndarray
        Positions of oxygen atoms.
    cell : ase.cell.Cell
        Simulation cell object defining periodic boundaries.
    oh_cutoff : float
        Maximum O-H distance to consider as a bond.

    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping oxygen atom indices to lists of two bonded hydrogen atom indices.
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
    water_dict: Dict[int, List[int]],
    cell: Cell,
) -> np.ndarray:
    """
    Calculate dipole moments for water molecules.

    Parameters
    ----------
    h_positions : np.ndarray
        Positions of hydrogen atoms.
    o_positions : np.ndarray
        Positions of oxygen atoms.
    water_dict : Dict[int, List[int]]
        Dictionary mapping oxygen atom indices to two bonded hydrogen atom indices.
    cell : ase.cell.Cell
        Simulation cell object defining periodic boundaries.

    Returns
    -------
    np.ndarray
        Array of dipole vectors for each oxygen atom. Entries are NaN for non-water oxygen atoms.
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
    Analyze water density and orientation near surfaces using MDAnalysis.

    Parameters
    ----------
    xyz : str
        Path to the input XYZ trajectory file.
    cell : ase.cell.Cell
        Simulation cell object defining periodic boundaries.
    surf1 : np.ndarray
        Atom indices defining the first surface.
    surf2 : np.ndarray
        Atom indices defining the second surface.
    **kwargs : dict, optional
        Keyword arguments:
        - `dt` : float
            Time step between frames in the trajectory.
        - `verbose` : bool
            Verbosity level for AnalysisBase.
        - `oh_cutoff` : float
            Maximum O-H bond length to define water molecules. Default is 2 Å.
        - `strict` : bool
            Strict handling of water molecules (identify water molecules again each timestep).
            Default is False.
        - `origin` : str
            Where to place the coordinate system origin ("surf1" for the first surface or "center"
            for the electrolyte center). Default is "surf1".
        - `dz` : float
            Bin size for density profiles in Å. Default is 0.1.
        - `oxygen_indices` : list[int]
            Indices of oxygen atoms. Default selects all atoms with name "O".
        - `hydrogen_indices` : list[int]
            Indices of hydrogen atoms. Default selects all atoms with name "H".

    Attributes
    ----------
    cell : ase.cell.Cell
        Simulation cell object.
    surf1_ag : AtomGroup
        Atom group for the first surface.
    surf2_ag : AtomGroup
        Atom group for the second surface.
    oh_cutoff : float
        Maximum O-H bond length.
    strict : bool
        Strict handling of water molecules.
    origin : str
        Coordinate system origin method.
    dz : float
        Bin size for density profiles.
    o_ag : AtomGroup
        Atom group for oxygen atoms.
    h_ag : AtomGroup
        Atom group for hydrogen atoms.
    water_dict : Dict[int, List[int]]
        Mapping of oxygen atom indices to two bonded hydrogen atom indices.
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
            dt=kwargs.get("dt"),
        )

        # Save required arguments
        self.cell = cell
        logger.info("Cell: %s", str(cell))
        self.surf1_ag = universe.select_atoms(*[f"index {i}" for i in surf1])
        self.surf2_ag = universe.select_atoms(*[f"index {i}" for i in surf2])
        logger.info("Surface 1 atom indices: %s", str(surf1))
        logger.info("Surface 2 atom indices: %s", str(surf2))

        # Initialize AnalysisBase
        super().__init__(universe.trajectory, verbose=kwargs.get("verbose", None))

        # Parse optional kwargs
        self.oh_cutoff = kwargs.get("oh_cutoff", 2)
        self.strict = kwargs.get("strict", False)
        logger.info("Strict handling of water molecules: strict=%s", str(self.strict))
        self.origin = kwargs.get("origin", "surf1")
        logger.info("Coordinate system origin: origin=%s", self.origin)
        self.dz = kwargs.get("dz", 0.1)
        logger.info("Bin size for density profiles: dz=%.2f", self.dz)

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

        # Define results objects
        self.n_frames = None
        self.results.z1 = None
        self.results.z2 = None
        self.results.z_water = None
        self.results.cos_theta = None
        self.results.dipole = None

    def get_origin(self, z1: float | np.ndarray, z2: float | np.ndarray) -> np.ndarray:
        """
        Determine the origin for the coordinate system.

        Parameters
        ----------
        z1 : float or np.ndarray
            Z-coordinate(s) of the first surface.
        z2 : float or np.ndarray
            Z-coordinate(s) of the second surface.

        Returns
        -------
        np.ndarray
            Z-coordinate(s) of the origin based on the specified reference (`"center"` or `"surf1"`).
            Defaults to an array of zeros if the origin is not recognized.
        """
        z1 = np.atleast_1d(z1)
        z2 = np.atleast_1d(z2)
        if self.origin == "center":
            return (z1 + z2) / 2
        if self.origin == "surf1":
            return z1
        return np.zeros(z1.shape)

    def _prepare(self):
        # At this point, n_frames is set by self._setup_frames in the base class
        # Initialize results
        self.results.z1 = np.zeros(self.n_frames)
        self.results.z2 = np.zeros(self.n_frames)
        self.results.z_water = np.zeros((self.n_frames, len(self.o_ag)))
        self.results.cos_theta = np.zeros((self.n_frames, len(self.o_ag)))
        self.results.dipole = np.zeros((self.n_frames, len(self.o_ag), 3))

    def _single_frame(self):
        """
        Compute surface position, water density and cos theta for a single frame
        """
        # Surface position
        z1 = np.mean(self.surf1_ag.positions[:, 2])
        z2 = np.mean(self.surf2_ag.positions[:, 2])
        self.results.z1[self._frame_index] = z1
        self.results.z2[self._frame_index] = z2 + self.cell[2][2] * (z1 > z2)

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

        # Set coordinate origin
        origin = self.get_origin(self.results.z1, self.results.z2)
        z_water = self.results.z_water - origin[:, np.newaxis]

        # Surface locations
        z1 = self.results.z1.mean() - origin.mean()
        z2 = self.results.z2.mean() - origin.mean()

        # Water density
        counts, bin_edges = np.histogram(
            z_water.flatten(),
            bins=int((z2 - z1) / self.dz),
            range=(z1, z2),
        )
        n_water = counts / self.n_frames
        grid_volume = np.diff(bin_edges) * area
        self.results.density_profile = [
            bin_edges_to_grid(bin_edges),
            water_density(n_water, grid_volume),
        ]

        # Water orientation
        valid = ~np.isnan(self.results.cos_theta.flatten())
        counts, bin_edges = np.histogram(
            z_water.flatten()[valid],
            bins=int((z2 - z1) / self.dz),
            range=(z1, z2),
            weights=self.results.cos_theta.flatten()[valid],
        )
        self.results.orientation_profile = [
            bin_edges_to_grid(bin_edges),
            counts / self.n_frames,
        ]

    def plot_orientation(self, ax: Axes, sym: bool = False) -> None:
        """
        Plot the orientation profile of water molecules.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on.
        sym : bool, optional
            If True, plots the symmetrized profile. Default is False.

        Returns
        -------
        None
        """
        x, y = self.results.orientation_profile
        if sym:
            y = (y - y[::-1]) / 2
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.5)
        ax.set_ylabel(r"$\rho \langle \cos \theta \rangle$ / g cm$^{-3}$")
        ax.set_xlabel(r"$z$ / Å")

    def plot_density(self, ax: Axes, sym: bool = False) -> None:
        """
        Plot the density profile of water molecules.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on.
        sym : bool, optional
            If True, plots the symmetrized profile. Default is False.

        Returns
        -------
        None
        """
        x, y = self.results.density_profile
        if sym:
            y = (y + y[::-1]) / 2
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.5)
        ax.set_ylabel(r"$\rho$ / g cm$^{-3}$")
        ax.set_xlabel(r"$z$ / Å")
