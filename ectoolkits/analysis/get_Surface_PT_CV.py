
# TODO: refactor to an analysis module
# test example: tests/analysis/pt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis import Universe
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_bonds
from ase.io import read

def get_candidates_hbonds(idxs_donor, idxs_acceptor, idxs_O_pair, idxs_H_pair):
    candidates_hbonds = []
    for idx_donor in idxs_donor:
        idxs_bonded_H = idxs_H_pair[idxs_O_pair == idx_donor]
        for idx_bonded_H in idxs_bonded_H:
            for idx_acceptor in idxs_acceptor:
                candidates_hbonds.append([idx_donor, idx_bonded_H, idx_acceptor])
    candidates_hbonds = np.array(candidates_hbonds)
    return candidates_hbonds


# PT group 1 and 2, can be donor or acceptor
idxs_type1_o = [150]
#idxs_type2_o = [130, 241, 136, 139, 129]

idxs_type2_o = [138, 131]

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
