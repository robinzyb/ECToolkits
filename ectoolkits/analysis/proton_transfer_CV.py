
# TODO: refactor to an analysis module
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


class ProtonTransferCV(AnalysisBase):
    def __init__(self,
                 atomgroup,
                 idxs_type1_o: List[int],
                 idxs_type2_o: List[int],
                 verbose=True,
                 **kwargs):
        """
        Set up the initial analysis parameters.
        I don't know the reason why the offical example always put atomgroup as the first argument.
        Universe class can also access the atomgroup and trajectory classes.
        """
        trajectory = atomgroup.universe.trajectory
        super(ProtonTransferCV, self).__init__(trajectory,
                                               verbose=verbose)

        self.u = atomgroup.universe

        self.idx_type1_o = idxs_type1_o
        self.idx_type2_o = idxs_type2_o


        # How should I select atoms inside AnalysisBase?
        # Now I just use the Universe class to select atoms.
        self.ag_O = self.u.select_atoms('type O')
        self.ag_H = self.u.select_atoms('type H')

        self.n_results = 9

    def _prepare(self):
        self.results = np.zeros((self.n_frames, self.n_results))

    def _single_frame(self):


        idxs_OH_pair = self._gen_connectivity(self)

        # type1 as donor and type2 as acceptor
        # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]
        candidates_hbonds = get_candidates_hbonds(idxs_type1_o, idxs_type2_o, idxs_OH_pair)

        # type2 as donor and type1 as acceptor
        # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]
        candidates_hbonds_2 = get_candidates_hbonds(idxs_type2_o, idxs_type1_o, idxs_OH_pair)

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
        # [frame, donor_index, hydrogen_index, acceptor_index, DA_distance, DAH_angle, delta, dOdHd, dOaHd]
        columns = ['Frame',
                   'Index of donor',
                   'Index of hydrogen',
                   'Index of acceptor',
                   'Interatomic distance (Donor-Aceeptor)',
                   'Angle (Donor-Acceptor-Hydrogen)',
                   'Delta',
                   'Distance (Donor-Hydrogen)',
                   'Distance (Acceptor-Hydrogen)'
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
        dist_array = distance_array(poses_H, poses_O, box=u.dimensions)
        idxs_O_pair = idxs_O[np.argmin(dist_array, axis=1)]
        idxs_H_pair = idxs_H
        idxs_OH_pair = np.stack((idxs_O_pair, idxs_H_pair), axis=1)
        return idxs_OH_pair

    def _get_candidates_hbonds(idxs_donor: List[int],
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
            poses_donor_o = u.atoms.positions[candidates_hbonds.T[0]]
            poses_bonded_H = u.atoms.positions[candidates_hbonds.T[1]]
            poses_acceptor_o = u.atoms.positions[candidates_hbonds.T[2]]

            # check if the h bond forms
            # angle < OaOdHd <= 30
            angles = calc_angles(poses_acceptor_o, poses_donor_o, poses_bonded_H, box=u.dimensions)/np.pi*180
            # dist OaOd < 3.5
            bonds = calc_bonds(poses_acceptor_o, poses_donor_o, box=u.dimensions)


            mask = (angles <= 30) & (bonds < 3.5)
            hbonds = candidates_hbonds[mask]

            # update the poses
            poses_type1_o = u.atoms.positions[hbonds.T[0]]
            poses_bonded_H = u.atoms.positions[hbonds.T[1]]
            poses_type2_o = u.atoms.positions[hbonds.T[2]]
            # calculate delta
            dOdHd = calc_bonds(poses_type1_o, poses_bonded_H, box=self.u.dimensions)
            dOaHd = calc_bonds(poses_type2_o, poses_bonded_H, box=self.u.dimensions)
            delta = dOdHd - dOaHd
            angles = angles[mask]
            bonds = bonds[mask]
            # information
            # [frame, donor_index, hydrogen_index, acceptor_index, DA_distance, DAH_angle, delta, dOdHd, dOaHd]
            if len(hbonds) == 0:
                info[1:] = np.nan
            else:
                tmp_info = np.zeros((len(hbonds), self.n_results))
                tmp_info[:, 0] = self._ts.frame
                tmp_info[:, 1] = hbonds.T[0]
                tmp_info[:, 2] = hbonds.T[1]
                tmp_info[:, 3] = hbonds.T[2]
                tmp_info[:, 4] = bonds
                tmp_info[:, 5] = angles
                tmp_info[:, 6] = delta
                tmp_info[:, 7] = dOdHd
                tmp_info[:, 8] = dOaHd

                info = tmp_info[np.argmin(info, axis=0)[6]]

            return info


def get_candidates_hbonds(idxs_donor, idxs_acceptor, idxs_O_pair, idxs_H_pair):
    candidates_hbonds = []
    for idx_donor in idxs_donor:
        idxs_bonded_H = idxs_H_pair[idxs_O_pair == idx_donor]
        for idx_bonded_H in idxs_bonded_H:
            for idx_acceptor in idxs_acceptor:
                candidates_hbonds.append([idx_donor, idx_bonded_H, idx_acceptor])
    candidates_hbonds = np.array(candidates_hbonds)
    return candidates_hbonds



if __name__ == "__main__":
    # PT group 1 and 2, can be donor or acceptor
    idxs_type1_o = [150]
    idxs_type2_o = [130, 241, 136, 139, 129]

    #idxs_type2_o = [138, 131]

    all_info = []

    for i in range(6):
        #lammpstrj = "clip.lammpstrj"
        lammpstrj = Path(f"../{(i*5):02d}-{((i+1)*5):02d}ns.dump.lammpstrj")
        print("Processing", lammpstrj, flush=True)
        stc = read(lammpstrj, format="lammps-dump-text", specorder=["Bi", "H", "O", "V"])
        chemical_symbols = stc.get_chemical_symbols()
        u= Universe(lammpstrj, format="LAMMPSDUMP")
        u.add_TopologyAttr("types", np.array(chemical_symbols))
        for ts in u.trajectory:
            # for each frame
            ag_O = u.select_atoms('type O')
            poses_O = ag_O.positions
            idxs_O = ag_O.indices
            ag_H = u.select_atoms('type H')
            poses_H = ag_H.positions
            idxs_H = ag_H.indices

            # For the analysis, we define each H atom to be “covalently bound” to its nearest O atom.
            dist_array = distance_array(poses_H, poses_O, box=u.dimensions)
            idxs_O_pair = idxs_O[np.argmin(dist_array, axis=1)]
            idxs_H_pair = idxs_H
            idxs_OH_pair = np.stack((idxs_O_pair, idxs_H_pair), axis=1)

            # type1 as donor and type2 as acceptor
            # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]

            candidates_hbonds = get_candidates_hbonds(idxs_type1_o, idxs_type2_o, idxs_O_pair, idxs_H_pair)

            # type2 as donor and type1 as acceptor
            # candidates_hbonds = [ (donor_idx, hydrogen_idx, acceptor_idx..)]

            candidates_hbonds_2 = get_candidates_hbonds(idxs_type2_o, idxs_type1_o, idxs_O_pair, idxs_H_pair)

            if (candidates_hbonds.size != 0) and (candidates_hbonds_2.size != 0):
                candidates_hbonds = np.concatenate((candidates_hbonds, candidates_hbonds_2), axis=0)
            elif (candidates_hbonds_2.size != 0) and (candidates_hbonds.size == 0):
                candidates_hbonds = candidates_hbonds_2
            elif (candidates_hbonds.size == 0) and (candidates_hbonds_2.size == 0):
                print("No hbond candidates found")
                info = np.zeros((9))
                info[0] = ts.frame
                info[1:] = np.nan
                print(info)
                continue


            poses_donor_o = u.atoms.positions[candidates_hbonds.T[0]]
            poses_bonded_H = u.atoms.positions[candidates_hbonds.T[1]]
            poses_acceptor_o = u.atoms.positions[candidates_hbonds.T[2]]

            # check if the h bond forms
            # angle < OaOdHd <= 30
            angles = calc_angles(poses_acceptor_o, poses_donor_o, poses_bonded_H, box=u.dimensions)/np.pi*180
            # dist OaOd < 3.5
            bonds = calc_bonds(poses_acceptor_o, poses_donor_o, box=u.dimensions)


            mask = (angles <= 30) & (bonds < 3.5)
            hbonds = candidates_hbonds[mask]

            # update the poses
            poses_type1_o = u.atoms.positions[hbonds.T[0]]
            poses_bonded_H = u.atoms.positions[hbonds.T[1]]
            poses_type2_o = u.atoms.positions[hbonds.T[2]]
            # calculate delta
            dOdHd = calc_bonds(poses_type1_o, poses_bonded_H, box=u.dimensions)
            dOaHd = calc_bonds(poses_type2_o, poses_bonded_H, box=u.dimensions)
            delta = dOdHd - dOaHd
            angles = angles[mask]
            bonds = bonds[mask]
            # information
            # [frame, donor_index, hydrogen_index, acceptor_index, DA_distance, DAH_angle, delta, dOdHd, dOaHd]
            if len(hbonds) == 0:
                info = np.zeros((9))
                info[0] = ts.frame
                info[1:] = np.nan
            else:
                info = np.zeros((len(hbonds), 9))
                info[:, 0] = ts.frame
                info[:, 1] = hbonds.T[0]
                info[:, 2] = hbonds.T[1]
                info[:, 3] = hbonds.T[2]
                info[:, 4] = bonds
                info[:, 5] = angles
                info[:, 6] = delta
                info[:, 7] = dOdHd
                info[:, 8] = dOaHd

                info = info[np.argmin(info, axis=0)[6]]

            # check the shape
            if info.shape[0] != 9:
                raise ValueError("The shape of info is not correct")
            #print(info)

            all_info.append(info)

    all_info = np.array(all_info)
    np.save("SurfacePT-CV-Osd.npy", all_info, allow_pickle=False)
