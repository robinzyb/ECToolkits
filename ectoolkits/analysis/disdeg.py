import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis import AtomGroup


def count_AB_CN(positions: np.array,
                A_idxs: List[int],
                B_idxs: List[int],
                box: List[float],
                max_cutoff: float = 1.2,
                min_cutoff: Optional[float] = None,
                ):
    """
    count the number of B atoms around each A atom.

    Args:
    -----------
        positions (np.array):
            The positions of all atoms, shape(N,3).
        A_idxs (List[int]):
            The indices of A atoms.
        B_idxs (List[int]):
            The indices of B atoms.
        box (List[float], optional):
            The box information. cellpar, dimension should be 6
            [a, b, c, alpha, beta, gamma]
        max_cutoff (float, optional):
            maximum cutoff of neighbors. Defaults to 1.2.
        min_cutoff (Optional[float], optional):
            minimum cutoff of neighbors. Defaults to None.

    Returns:
    -----------
        cn_list (np.array):
            list of coordination numbers of A atoms, shape(length of A_idx,).

    Notes:
    -----------
     _notes_

    Examples:
    -----------
    ```python
    >>> from MDAnalysis import Universe
    >>> from ase.io import read
    >>> xyzfile = "./tio2-water.xyz"
    >>> stc_ase = read(xyzfile)
    >>> cellpar = stc_ase.get_cell().cellpar()
    >>> u = Universe(xyzfile)
    >>> u.dimensions = cellpar
    >>> from ectoolkits.analysis.disdeg import count_AB_CN
    >>> u.atoms.positions
    array([[ 2.7988806,  5.6870356,  1.9968334],
       [ 2.7716577,  2.41441  ,  5.3606954],
       [ 2.7103157,  5.5256004,  8.89024  ],
       ...,
       [ 5.2708693,  2.7994716, 30.055622 ],
       [ 5.75261  ,  3.4865937, 29.678984 ],
       [ 4.6656394,  3.161835 , 30.73101  ]], dtype=float32)
    >>> O_idx = [264, 267, 270, 273, 276, 279, 282, 285]
    >>> H_idx = np.where(u.atoms.elements == "H")[0]
    >>> box = u.dimensions
    >>> print("O indices", O_idx)
    O indices [264, 267, 270, 273, 276, 279, 282, 285]
    >>> print("H indices", H_idx)
    H indices [241 242 244 245 247 248 250 251 253 254 256 257 259 260 262 263 265 266
    268 269 271 272 274 275 277 278 280 281 283 284 286 287 289 290 292 293
    295 296 298 299 301 302 304 305 307 308 310 311 313 314 316 317 319 320
    322 323 325 326 328 329 331 332 334 335 337 338 340 341 343 344 346 347
    349 350 352 353 355 356 358 359 361 362 364 365 367 368 370 371 373 374
    376 377 379 380 382 383 385 386 388 389 391 392 394 395 397 398 400 401
    403 404 406 407 409 410 412 413 415 416 418 419 421 422 424 425 427 428
    430 431 433 434 436 437 439 440 442 443 445 446 448 449 451 452 454 455
    457 458 460 461 463 464 466 467 469 470 472 473 475 476 478 479 481 482
    484 485 487 488 490 491 493 494 496 497 499 500 502 503 505 506 508 509
    511 512 514 515 517 518 520 521 523 524 526 527 529 530 532 533 535 536
    538 539 541 542 544 545 547 548 550 551 553 554 556 557 559 560 562 563
    565 566 568 569 571 572 574 575 577 578 580 581 583 584 586 587 589 590
    592 593 595 596 598 599 601 602]
    >>> print("box", box)
    box [11.836 12.994 38.53  90.    90.    90.   ]
    >>> cn_list = count_AB_CN(u.atoms.positions, O_idx, H_idx, box, 1.2)
    >>> print("coordination number for O_idx is ", cn_list)
    coordination number for O_idx is  [2 2 2 2 2 2 1 2]
    ```
    """

    _pairs = capped_distance(positions[A_idxs],
                             positions[B_idxs],
                             box=box,
                             min_cutoff=min_cutoff,
                             max_cutoff=max_cutoff,
                             return_distances=False,
                             )

    _pairs = _pairs[:, 0]

    return np.bincount(_pairs, minlength=len(A_idxs))


def cumsum_arr(arr):
    # not vectorize:
    # seem same in ectoolkits utils
    cumulative_sum = np.cumsum(arr)
    for i, _num in enumerate(cumulative_sum):
        cumulative_sum[i] = cumulative_sum[i]/(i+1)
    return cumulative_sum


class CNState(AnalysisBase):
    """
    Count Coordination number for a given index list in a trajectory.
    """
    _cnstate = None

    def __init__(self,
                 atomgroup: AtomGroup = None,
                 center_atom_idx: np.array = None,
                 coordinated_elements: List[str] = ["H"],
                 max_cutoff: float = 1.2,
                 min_cutoff: float = None,
                 ):
        """
        Initialize CNState class.

        Args:
        -----------
            atomgroup (AtomGroup):
                AtomGroup object in MDAnalysis.
            center_atom_idxs (np.array):):
                Atom indices for center atoms.
                could be an array of atom indices of 1d
                or a 2d array of atom indices with shape of (n_frames, n_idx).
            coordinated_elements (List[str], optional):
                a list of elements as coordination of center atoms. Defaults to ["H"].
            max_cutoff (float, optional):
                . Defaults to 1.2.
            min_cutoff (float, optional):
                . Defaults to None.

        Raises:
        -----------
            ValueError:
                _description_

        Notes:
        -----------
         _notes_

        Examples:
        -----------
        ```python
        #
        >>> O_idx_dw = [264, 267, 270, 273, 276, 279, 282, 285]
        >>> cnstate = CNState(atomgroup=u.atoms,
        >>>          center_atom_idx=np.array(O_idx_dw),
        >>>          coordinated_elements=['H'],
        >>>          max_cutoff=1.2)
        >>> cnstate.run()
        >>> cnstate._cnstate
        array([[2, 2, 2, 2, 2, 2, 1, 2],
       [2, 2, 2, 2, 2, 2, 1, 2],
       [2, 2, 2, 2, 2, 2, 1, 2],
       [2, 2, 2, 2, 2, 2, 1, 2],
       [2, 2, 2, 2, 2, 2, 1, 2]])
        >>> cnstate.plot()

        # save the results to npy file
        cnstate.save_cnstate("cnstate.npy")
        ```
        """
        # ---------------------- intput parameters --------------------------- #
        self._ag = atomgroup
        self.center_atom_idx = center_atom_idx
        self.coordinated_elements = coordinated_elements
        self.max_cutoff = max_cutoff
        self.min_cutoff = min_cutoff

        # ----------------------- extra parameters --------------------------- #
        if self._ag is not None:
            self._trajectory = self._ag.universe.trajectory
            self.cellpar = self._ag.universe.dimensions

    @classmethod
    def read_cnstate_from(cls, npyfile: str):
        cls._cnstate = np.load(npyfile)
        return cls(atomgroup=None,
                   center_atom_idx=None,
                   coordinated_elements=None,
                   max_cutoff=None,
                   min_cutoff=None,)

    def _prepare(self):

        # turn a list of coordinated elements to a list of indices
        _elements = self._ag.atoms.elements
        if type(self.coordinated_elements) == str:
            self.coordinated_elements = [self.coordinated_elements]
        sel = np.empty(len(_elements), dtype=bool)
        sel.fill(0)
        for element in self.coordinated_elements:
            sel = (sel) | (_elements == element)
        self.coord_idx = np.where(sel)

        # ----------------------- assertion ---------------------------------- #
        if not (np.any(self.cellpar)):
            raise ValueError("Cell Information Needed!")
        assert self.cellpar is not None, \
            ("please provide cell information for Universe.dimensions")
        assert isinstance(self.center_atom_idx, np.ndarray), \
            ("center_atom_idx should be a 1d or 2d numpy array")

        # check if the center atom index is a 2d array or 1d array

        if self.center_atom_idx.ndim == 1:
            self.center_atom_idx = np.tile(self.center_atom_idx,
                                           (self.n_frames, 1),
                                           )

        assert self.center_atom_idx.ndim == 2, \
            ("center_atom_idx should be a 1d or 2d array")
        assert self.center_atom_idx.shape[0] == self.n_frames, \
            ("center_atom_idx should have "
             "the same number of frames as the trajectory"
             "you may provide center_atom_idx with a wrong shape"
             "or you can re-instantiate the class")

        self.num_center_atom_idx = self.center_atom_idx.shape[1]

        # place holder for the cnstate
        self._cnstate = np.zeros(
            (self.n_frames, self.num_center_atom_idx),
            dtype=int
        )

    def _single_frame(self):

        pos = self._ag.positions
        self._cnstate[self._frame_index] = count_AB_CN(pos,
                                                       self.center_atom_idx[self._frame_index],
                                                       self.coord_idx,
                                                       self.cellpar,
                                                       max_cutoff=self.max_cutoff,
                                                       min_cutoff=self.min_cutoff)

    def _conclude(self):
        pass

    def save_cnstate(self, filename):
        np.save(filename, self._cnstate)

    def get_cnstate_percentage(self, expected_cn: int):

        # make sure self._cnstate is 2d array
        # and self._cnstate exists
        assert self._cnstate.ndim == 2, \
            "cnstate shape is not 2d or not read from npy file"
        return np.count_nonzero(self._cnstate == expected_cn, axis=1)/self._cnstate.shape[1]

    def plot_cnstate(self, cn_list=[0, 1, 2, 3]):
        fig, ax = plt.subplots(figsize=(12, 3), dpi=500)
        for cn in cn_list:
            sns.lineplot(self.get_cnstate_percentage(
                expected_cn=cn), ax=ax, label=f"CN={cn}")
        return fig


class DisDeg(CNState):
    """
    Class for calculating the degree of dissociation,
    Child Class of CNState

    Args:
    -----------
        CNState (_type_):
            _description_

    Notes:
    -----------
     _notes_

    Examples:
    -----------
     _examples_
    """

    def _conclude(self):
        self.get_disdeg()

    @classmethod
    def read_cnstate_from(cls, npyfile: str):
        cls._cnstate = np.load(npyfile)
        return cls(atomgroup=None,
                   center_atom_idx=None,
                   coordinated_elements=None,
                   max_cutoff=None,
                   min_cutoff=None,)

    def get_disdeg(self,
                   cn_list_no_dis: List[int] = [2, 3],
                   cn_list_dis: List[int] = [0, 1]
                   ):

        self._no_disdeg = np.zeros((self._cnstate.shape[0]), dtype=float)
        self._disdeg = np.zeros((self._cnstate.shape[0]), dtype=float)
        for cn in cn_list_no_dis:
            self._no_disdeg += self.get_cnstate_percentage(expected_cn=cn)
        for cn in cn_list_dis:
            self._disdeg += self.get_cnstate_percentage(expected_cn=cn)

    def save_disdeg(self, filename: str):
        np.save(filename, [self._disdeg])

    def plot_disdeg(self):

        fig, ax = plt.subplots(figsize=(12, 3), dpi=500)
        sns.lineplot(self._disdeg, ax=ax, label="disdeg")
        sns.lineplot(self._no_disdeg, ax=ax, label="no_disdeg")

        return fig

    # def save_obj(self, filename="disdeg.pkl", path_output="./"):

    #     import pickle
    #     with open(os.path.join(path_output, filename), "wb") as f:
    #         pickle.dump(self,f)
