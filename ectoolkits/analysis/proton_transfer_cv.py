# test example: tests/analysis/pt
# https://userguide.mdanalysis.org/stable/examples/analysis/custom_trajectory_analysis.html
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis import Universe
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_bonds
from ase.io import read


# TODO: refactor the indirect transfer together.
# TODO: pytest
class ProtonTransferCV(AnalysisBase):
    def __init__(self,
                 atomgroup,
                 idxs_type1_o: List[int],
                 idxs_type2_o: List[int],
                 num_bridge: int = 0,
                 verbose=True,
                 **kwargs):
        """
        num_bridge: int
            The number of bridge water molecules between the donor and acceptor.
        Set up the initial analysis parameters.
        I don't know the reason why the offical example always put atomgroup as the first argument.
        Universe class can also access the atomgroup and trajectory classes.
        """
        trajectory = atomgroup.universe.trajectory
        super(ProtonTransferCV, self).__init__(trajectory,
                                               verbose=verbose)

        self.u = atomgroup.universe

        self.idxs_type1_o = idxs_type1_o
        self.idxs_type2_o = idxs_type2_o


        # How should I select atoms inside AnalysisBase?
        # Now I just use the Universe object to select atoms.
        self.ag_O = self.u.select_atoms('type O')
        self.ag_H = self.u.select_atoms('type H')

        self.n_results = 10

    def _prepare(self):
        self.results = np.zeros((self.n_frames, self.n_results))

    def _single_frame(self):


        idxs_OH_pair = self._gen_connectivity()

        # type1 as donor and type2 as acceptor
        # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]
        candidates_hbonds = self._get_candidates_hbonds(self.idxs_type1_o, self.idxs_type2_o, idxs_OH_pair)

        # type2 as donor and type1 as acceptor
        # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]
        candidates_hbonds_2 = self._get_candidates_hbonds(self.idxs_type2_o, self.idxs_type1_o, idxs_OH_pair)

        # merge the two found candidates_hbonds
        if candidates_hbonds.size == 0:
            candidates_hbonds = candidates_hbonds_2
        elif candidates_hbonds_2.size != 0:
            candidates_hbonds = np.concatenate((candidates_hbonds, candidates_hbonds_2), axis=0)

        info = self._get_proton_transfer_cv(candidates_hbonds)
        self.results[self._frame_index] = info


    def _conclude(self):
        """
        Finish up by calculating an average and transforming our
        results into a DataFrame.
        """
        # by now self.result is fully populated
        # [frame, delta_cv, distance_cv, donor_index, hydrogen_index, acceptor_index, dOdHd, dOaHd, DA_distance, DAH_angle ]
        columns = ['Frame',
                   'Delta_CV',
                   'Distance_CV',
                   'Index of donor',
                   'Index of hydrogen',
                   'Index of acceptor',
                   'Distance (Donor-Hydrogen)',
                   'Distance (Acceptor-Hydrogen)',
                   'Interatomic distance (Donor-Aceeptor)',
                   'Angle (Donor-Acceptor-Hydrogen)',
                   ]
        self.df = pd.DataFrame(self.results, columns=columns)

    # helper functions
    def _gen_connectivity(self):
        """
        Generate the connectivity between O and H atoms.
        """
        poses_O = self.ag_O.positions
        idxs_O = self.ag_O.indices
        poses_H = self.ag_H.positions
        idxs_H = self.ag_H.indices
        # We define each H atom to be “covalently bound” to its nearest O atom.
        # See 10.1021/acs.jpclett.7b00358
        dist_array = distance_array(poses_H, poses_O, box=self.u.dimensions)
        idxs_O_pair = idxs_O[np.argmin(dist_array, axis=1)]
        idxs_H_pair = idxs_H
        idxs_OH_pair = np.stack((idxs_O_pair, idxs_H_pair), axis=1)
        return idxs_OH_pair

    def _get_candidates_hbonds(self,
                               idxs_donor: List[int],
                               idxs_acceptor: List[int],
                               idxs_OH_pair: npt.NDArray,
                               ):
        """
        Get the candidates of hydrogen bonds.
        """
        idxs_O_pair = idxs_OH_pair[:, 0]
        idxs_H_pair = idxs_OH_pair[:, 1]

        candidates_hbonds = []
        for idx_donor in idxs_donor:
            idxs_bonded_H = idxs_H_pair[idxs_O_pair == idx_donor]
            for idx_bonded_H in idxs_bonded_H:
                for idx_acceptor in idxs_acceptor:
                    candidates_hbonds.append([idx_donor, idx_bonded_H, idx_acceptor])
        candidates_hbonds = np.array(candidates_hbonds)
        return candidates_hbonds

    def _get_proton_transfer_cv(self, candidates_hbonds):
        info = np.zeros((self.n_results))
        info[0] = self._ts.frame
        if candidates_hbonds.size == 0:
            info[1:] = np.nan
        else:
            poses_donor_o = self.u.atoms.positions[candidates_hbonds.T[0]]
            poses_bonded_H = self.u.atoms.positions[candidates_hbonds.T[1]]
            poses_acceptor_o = self.u.atoms.positions[candidates_hbonds.T[2]]

            # check if the h bond forms
            # angle < OaOdHd <= 30
            angles = calc_angles(poses_acceptor_o, poses_donor_o, poses_bonded_H, box=self.u.dimensions)/np.pi*180
            # dist OaOd < 3.5
            bonds = calc_bonds(poses_acceptor_o, poses_donor_o, box=self.u.dimensions)


            mask = (angles <= 30) & (bonds < 3.5)
            hbonds = candidates_hbonds[mask]

            # update the poses
            poses_type1_o = self.u.atoms.positions[hbonds.T[0]]
            poses_bonded_H = self.u.atoms.positions[hbonds.T[1]]
            poses_type2_o = self.u.atoms.positions[hbonds.T[2]]
            # calculate delta
            dOdHd = calc_bonds(poses_type1_o, poses_bonded_H, box=self.u.dimensions)
            dOaHd = calc_bonds(poses_type2_o, poses_bonded_H, box=self.u.dimensions)
            delta = dOdHd - dOaHd
            angles = angles[mask]
            bonds = bonds[mask]
            # information
            # [frame, delta_cv, distance_cv, donor_index, hydrogen_index, acceptor_index, dOdHd, dOaHd, DA_distance, DAH_angle ]
            if len(hbonds) == 0:
                info[1:] = np.nan
            else:
                tmp_info = np.zeros((len(hbonds), self.n_results))
                tmp_info[:, 0] = self._ts.frame
                tmp_info[:, 1] = delta
                tmp_info[:, 2] = bonds
                tmp_info[:, 3] = hbonds.T[0]
                tmp_info[:, 4] = hbonds.T[1]
                tmp_info[:, 5] = hbonds.T[2]
                tmp_info[:, 6] = dOdHd
                tmp_info[:, 7] = dOaHd
                tmp_info[:, 8] = bonds
                tmp_info[:, 9] = angles

                # TODO: change the sign of delta_cv
                # the delta is always negative here because it is dOdHd - dOaHd.
                # for donor in type2, acceptor in type1, the delta should be positive.

                info = tmp_info[np.argmin(tmp_info, axis=0)[1]]

            return info



