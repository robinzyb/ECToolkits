# test example: tests/analysis/proton_transfer_cv
# https://userguide.mdanalysis.org/stable/examples/analysis/custom_trajectory_analysis.html
from pathlib import Path
from typing import List, Tuple
from itertools import product, permutations

import numpy as np
import numpy.typing as npt
import pandas as pd
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_bonds

from ectoolkits.log import get_logger

logger = get_logger(__name__)

class ProtonTransferCV(AnalysisBase):
    """
    Class for analyzing Proton Transfer Collective Variables (CVs) from molecular dynamics simulations.

    This class extends the MDAnalysis `AnalysisBase` class and provides methods to analyze proton transfer
    events between donor and acceptor atoms, potentially mediated by bridge water molecules.

    Attributes:
        u (Universe): The MDAnalysis Universe object containing the atomgroup and trajectory.
        idxs_type1_o (List[int]): List of indices for type 1 oxygen atoms (donors).
        idxs_type2_o (List[int]): List of indices for type 2 oxygen atoms (acceptors).
        num_bridge (int): The number of bridge water molecules between the donor and acceptor.
        idxs_water_o (List[int]): List of indices for bridge water oxygen atoms.
        ag_O (AtomGroup): AtomGroup of oxygen atoms.
        ag_H (AtomGroup): AtomGroup of hydrogen atoms.
        n_results (int): Number of results to store for each frame.
        results (ndarray): Array to store the results of the analysis.
        extra_detail (bool): Whether to include detailed results in the analysis. Defaults to True.

    Methods:
        _prepare(): Prepares the results array for storing analysis results.
        _single_frame(): Analyzes a single frame of the trajectory.
        _conclude(): Finalizes the analysis by calculating averages and transforming results into a DataFrame.
        _gen_connectivity(): Generates the connectivity between oxygen and hydrogen atoms.
        _get_candidates_hbonds(): Identifies candidate hydrogen bonds based on donor, acceptor, and bridge atoms.
        _get_proton_transfer_cv(): Calculates the proton transfer collective variables for candidate hydrogen bonds.
    """
    def __init__(self,
                 atomgroup,
                 idxs_type1_o: List[int],
                 idxs_type2_o: List[int],
                 num_bridge: int = 0,
                 verbose=True,
                 **kwargs):
        """
        Initialize the ProtonTransferCV analysis.

        Parameters:
            atomgroup (AtomGroup): The atomgroup to analyze.
            idxs_type1_o (List[int]): List of indices for type 1 oxygen atoms (donors).
            idxs_type2_o (List[int]): List of indices for type 2 oxygen atoms (acceptors).
            num_bridge (int): The number of bridge water molecules between the donor and acceptor. Defaults to 0.
            verbose (bool): Whether to print verbose output. Defaults to True.
            **kwargs: Additional keyword arguments.


        Set up the initial analysis parameters.
        I don't know the reason why the offical example always put atomgroup as the first argument.
        Universe class can also access the atomgroup and trajectory classes.
        """
        trajectory = atomgroup.universe.trajectory
        super(ProtonTransferCV, self).__init__(trajectory,
                                               verbose=verbose)

        logger.info("Analysis of Proton Transfer Collective Variables")

        self.u = atomgroup.universe

        self.idxs_type1_o = idxs_type1_o
        self.idxs_type2_o = idxs_type2_o

        self.num_bridge = num_bridge
        logger.info(f"Number of bridge water molecules: {self.num_bridge}")
        if self.num_bridge == 1:
            self.idxs_water_o = kwargs.get("idxs_water_o", [])
            if len(self.idxs_water_o) == 0:
                logger.warning("You set number of bridge water larger than 0, but no bridge water molecule is defined.")
        elif self.num_bridge == 0:
            self.idxs_water_o = []
        else:
            raise ValueError("The number of bridge water molecules should be 0 or 1.")



        # How should I select atoms inside AnalysisBase?
        # Now I just use the Universe object to select atoms.
        self.ag_O = self.u.select_atoms('type O')
        self.ag_H = self.u.select_atoms('type H')


        # number of results to store
        # frame, delta_cv distance_cv [3]
        # donor_index_0, hydrogen_index_0, donor_index_1, hydrogen_index_1,
        # ..., donor_index_num_bridge, hydrogen_index_num_bridge,
        # acceptor_index [(num_bridge+1)*2]
        # dOdHd_0, dOdHd_1, ..., dOdHd_num_bridge, [(num_bridge+1)]
        # dOaOd_0, dOaOd_1, ..., dOaOd_num_bridge [(num_bridge+1)]
        # DA_distance_0, DA_distance_1, ..., DA_distance_num_bridge [(num_bridge+1)]
        # DAH_angle_0, DAH_angle_1, ..., DAH_angle_num_bridge [(num_bridge+1)]
        # in total
        # 3 + (num_bridge+1)*2+1 + (num_bridge+1)*2 + (num_bridge+1) + (num_bridge+1)

        self.extra_detail = kwargs.get("extra_detail", True)
        if self.extra_detail:
            logger.info(f"You have set extra_detail to {self.extra_detail}, the results will include the detailed information.")
        else:
            logger.info(f"You have set extra_detail to {self.extra_detail}, the results will only include the collective variables.")

        if self.extra_detail:
            self.n_results = 4+(self.num_bridge+1)*6
        else:
            self.n_results = 3


    def _prepare(self):
        self.results = np.zeros((self.n_frames, self.n_results))

    def _single_frame(self):
        idxs_OH_pair = self._gen_connectivity()
        # filter bridge water
        # type1 as donor and type2 as acceptor
        # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]
        candidates_hbonds = self._get_candidates_hbonds(self.idxs_type1_o, self.idxs_type2_o, self.idxs_water_o, idxs_OH_pair, num_bridge=self.num_bridge)

        # type2 as donor and type1 as acceptor
        # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]
        candidates_hbonds_2 = self._get_candidates_hbonds(self.idxs_type2_o, self.idxs_type1_o, self.idxs_water_o, idxs_OH_pair, num_bridge=self.num_bridge)

        # merge the two found candidates_hbonds
        if candidates_hbonds.size == 0:
            candidates_hbonds = candidates_hbonds_2
        elif candidates_hbonds_2.size != 0:
            candidates_hbonds = np.concatenate((candidates_hbonds, candidates_hbonds_2), axis=0)

        info = self._get_proton_transfer_cv(candidates_hbonds, num_bridge=self.num_bridge)


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
                   ]

        if self.extra_detail:
            for i in range(self.num_bridge+1):
                columns += [f'Index of donor{i}', f'Index of hydrogen{i}']
            columns += ['Index of acceptor']
            for i in range(self.num_bridge+1):
                columns += [f'Distance (Donor{i}-Hydrogen{i})']
            for i in range(self.num_bridge+1):
                columns += [f'Distance (Acceptor{i}-Hydrogen{i})']
            for i in range(self.num_bridge+1):
                columns += [f'Interatomic distance (Donor{i}-Aceeptor{i})']
            for i in range(self.num_bridge+1):
                columns += [f'Angle (Donor{i}-Aceeptor{i}-Hydrogen{i})']

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
                               idxs_bridge_candidate: List[int],
                               idxs_OH_pair: npt.NDArray,
                               num_bridge: int = 0,
                               ):
        """
        Get the candidates of hydrogen bonds.
        """

        idxs_O_pair = idxs_OH_pair[:, 0]
        idxs_H_pair = idxs_OH_pair[:, 1]

        # fisrt make product of donor and acceptor
        # insert the permuation of bridge water oxygen.
        # and make the candidates_hbonds_list in the order of donor0, hydrogen0, donor1, hydrogen1, ..., acceptor
        candidates_hbonds = []
        if num_bridge == 0:

            for idx_donor, idx_acceptor in product(idxs_donor, idxs_acceptor):
                idxs_bonded_H = idxs_H_pair[idxs_O_pair == idx_donor]
                for idx_bonded_H in idxs_bonded_H:
                    candidates_hbonds.append([idx_donor, idx_bonded_H, idx_acceptor])

        # TODO: generalize to arbitrary number of bridge water molecules, I should consider how to make a list of attached hydrogen atoms,
        # now I only refactor old indirect transfer code to here..
        elif num_bridge == 1:
            for idx_donor, idx_acceptor, idxs_bridge in product(idxs_donor, idxs_acceptor, permutations(idxs_bridge_candidate, num_bridge)):

                candidates_donor = [idx_donor]+list(idxs_bridge)
                # candidates_donor should be 2 elements
                # donor O , H1, bridge O , H2, acceptor O
                idx_type1 = candidates_donor[0]
                idxs_bonded_H1 = idxs_H_pair[idxs_O_pair == idx_type1]
                idx_bridge = candidates_donor[1]
                idxs_bonded_H2 = idxs_H_pair[idxs_O_pair == idx_bridge]
                # type1 is donor and type 2 is acceptor
                # the following can be refactored to permuation..
                for idx_bonded_H1 in idxs_bonded_H1:
                    for idx_bonded_H2 in idxs_bonded_H2:
                        candidates_hbonds.append([idx_type1, idx_bonded_H1, idx_bridge, idx_bonded_H2, idx_acceptor])
        else:
            raise ValueError("The number of bridge water molecules should be 0 or 1.")

        return np.array(candidates_hbonds)

    def _get_proton_transfer_cv(self, candidates_hbonds, num_bridge: int = 0):

        # hbond standard
        hbond_dOO = 3.5
        hbond_OaOdH_angle = 30

        n_all_results = 4+(self.num_bridge+1)*6
        # this info includes everything and will be tailored at the return statement.
        info = np.zeros((n_all_results))
        info[0] = self._ts.frame

        if candidates_hbonds.shape[0] == 0:
            info[1:] = np.nan
        else:
            # check if the h bond forms
            # check hbond distance first
            # donor_index_0, hydrogen_index_0, donor_index_1, hydrogen_index_1,
            # ..., donor_index_num_bridge, hydrogen_index_num_bridge, acceptor_index
            num_candidates = candidates_hbonds.shape[0]
            mask = np.full(num_candidates, True)
            for i in range(0, num_bridge*2+2, 2):
                poses_donor_o = self.u.atoms.positions[candidates_hbonds.T[i]]
                poses_acceptor_o = self.u.atoms.positions[candidates_hbonds.T[i+2]]
                bonds = calc_bonds(poses_acceptor_o, poses_donor_o, box=self.u.dimensions)
                mask = mask & (bonds < hbond_dOO)

            candidates_hbonds = candidates_hbonds[mask]


            # then check the angle
            num_candidates = candidates_hbonds.shape[0]
            mask = np.full(num_candidates, True)
            for i in range(0, num_bridge*2+2, 2):
                poses_donor_o = self.u.atoms.positions[candidates_hbonds.T[i]]
                poses_bonded_H = self.u.atoms.positions[candidates_hbonds.T[i+1]]
                poses_acceptor_o = self.u.atoms.positions[candidates_hbonds.T[i+2]]
                angles = calc_angles(poses_acceptor_o, poses_donor_o, poses_bonded_H, box=self.u.dimensions)/np.pi*180
                mask = mask & (angles <= hbond_OaOdH_angle)

            # All the candidates have hydrogen bond formed.
            hbonds = candidates_hbonds[mask]
            all_dOdHd = []
            all_dOaHd = []
            all_bonds = []
            all_angles = []
            for i in range(0, num_bridge*2+2, 2):
                poses_donor_o = self.u.atoms.positions[hbonds.T[i]]
                poses_bonded_H = self.u.atoms.positions[hbonds.T[i+1]]
                poses_acceptor_o = self.u.atoms.positions[hbonds.T[i+2]]
                # calculate delta
                dOdHd = calc_bonds(poses_donor_o, poses_bonded_H, box=self.u.dimensions)
                dOaHd = calc_bonds(poses_acceptor_o, poses_bonded_H, box=self.u.dimensions)
                bonds = calc_bonds(poses_acceptor_o, poses_donor_o, box=self.u.dimensions)
                angles = calc_angles(poses_acceptor_o, poses_donor_o, poses_bonded_H, box=self.u.dimensions)/np.pi*180

                # the shape of following is  (num_bridge+1, len(hbonds)
                all_dOdHd.append(dOdHd)
                all_dOaHd.append(dOaHd)
                all_bonds.append(bonds)
                all_angles.append(angles)

            all_dOdHd = np.array(all_dOdHd)
            all_dOaHd = np.array(all_dOaHd)
            all_bonds = np.array(all_bonds)
            all_angles = np.array(all_angles)

            delta_cv = np.mean(all_dOdHd - all_dOaHd, axis=0)
            distance_cv = np.mean(all_bonds, axis=0)

            # information
            # [frame, delta_cv, distance_cv, donor_index, hydrogen_index, acceptor_index, dOdHd, dOaHd, DA_distance, DAH_angle ]
            if len(hbonds) == 0:
                info[1:] = np.nan
            else:
                tmp_info = np.zeros((len(hbonds), n_all_results))
                tmp_info[:, 0] = self._ts.frame
                tmp_info[:, 1] = delta_cv
                tmp_info[:, 2] = distance_cv

                # store the hbond indexs
                tmp_info[:, 3:(6+2*num_bridge)] = hbonds

                # store the dOdHd, info

                tmp_info[:, (6+2*num_bridge):(7+3*num_bridge)] = all_dOdHd.T
                tmp_info[:, (7+3*num_bridge):(8+4*num_bridge)] = all_dOaHd.T
                tmp_info[:, (8+4*num_bridge):(9+5*num_bridge)] = all_bonds.T
                tmp_info[:, (9+5*num_bridge):(10+6*num_bridge)] = all_angles.T

                # 3 + (n_bridge+1)*2+1 + (n_bridge+1)*2 + (n_bridge+1)*2
                #
                # I take the one with a minimum value of |delta|.
                # It means the most reactive proton is chosen
                # refdoi: 10.1038/17579

                info = tmp_info[np.argmin(np.abs(tmp_info), axis=0)[1]]
                # the delta_cv is always negative here because it is dOdHd - dOaHd.
                # for donor in type2, acceptor in type1, the delta should be positive.
                # here I just check the donor index.
                if info[3] in self.idxs_type2_o:
                    info[1] = -info[1]



        if self.extra_detail is False:
            info = info[:3]

        return info



