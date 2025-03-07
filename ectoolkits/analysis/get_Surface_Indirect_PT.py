# for one water mediated PT
# find calculate distance between Oa and Om and Od and Om, find the oxygen sitting between Oa  and Od
# then find the H bond.

# TODO: refactor to an analysis module
# test example: tests/analysis/pt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis import Universe
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_bonds
from ase.io import read


# hbond standard
hbond_dOO = 3.5
hbond_OaOdH_angle = 30
# PT group 1 and 2, can be donor or acceptor
idxs_type1_o = [150]
idxs_type2_o = [138, 131]
#idxs_type2_o = [130, 241, 136, 139, 129]
idxs_water = [144, 151, 157, 162, 168, 169, 180, 183, 185, 186, 187, 188, 189, 191, 216, 217, 222, 223, 224, 226, 227, 228, 229, 232, 233, 234, 235, 236, 239, 240, 242, 243, 244, 247, 315, 321, 324, 327, 330, 339, 342, 345, 348, 351, 354, 130, 145, 156, 163, 174, 175, 181, 182, 184, 190, 218, 219, 220, 221, 225, 230, 231, 237, 238, 245, 246, 312, 318, 333, 336, 357]




# start of program
idxs_type1_o = np.array(idxs_type1_o)
idxs_type2_o = np.array(idxs_type2_o)
idxs_water = np.array(idxs_water)


def get_hbond_dist_angle(poses_donor_o,  poses_bonded_H, poses_acceptor_o, cell_param):
    angles = calc_angles(poses_acceptor_o, poses_donor_o, poses_bonded_H, box=cell_param)/np.pi*180
        # dist OaOd < 3.5
    bonds = calc_bonds(poses_acceptor_o, poses_donor_o, box=cell_param)
    return bonds, angles


if __name__ == "__main__":
    all_info = []
    for i in range(6):
        #lammpstrj = "clip_new.lammpstrj"
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


            # get the distance between Od and Ow

            poses = u.atoms.positions
            dist_array_OdOw = distance_array(poses[idxs_type1_o], poses[idxs_water], box=u.dimensions)
            dist_array_OaOw = distance_array(poses[idxs_type2_o], poses[idxs_water], box=u.dimensions)


            candidates_o_pathway = []
            # idxs_type1_o and idxs_type2_o should not overlap
            for i in range(len(idxs_type1_o)):
                for j in range(len(idxs_type2_o)):
                    mask = (dist_array_OdOw[i] < hbond_dOO) & (dist_array_OaOw[j] < hbond_dOO)
                    idxs_mediated_water = idxs_water[mask]
                    for idx_mediated_water in idxs_mediated_water:
                        candidates_o_pathway.append([idxs_type1_o[i], idx_mediated_water,  idxs_type2_o[j]])


            # generate candidates hbond list ( candidate for PT) from the candidates_o_pathway
            candidates_hbonds = []
            for candidate_o_pathway in candidates_o_pathway:
                # donor O , H1, mediated O , H2, acceptor O
                idx_type1 = candidate_o_pathway[0]
                idxs_bonded_H1 = idxs_H_pair[idxs_O_pair == idx_type1]
                idx_mediated = candidate_o_pathway[1]
                idxs_bonded_H2 = idxs_H_pair[idxs_O_pair == idx_mediated]
                idx_type2 = candidate_o_pathway[2]
                idxs_bonded_H3 = idxs_H_pair[idxs_O_pair == idx_type2]
                # type1 is donor and type 2 is acceptor
                for idx_bonded_H1 in idxs_bonded_H1:
                    for idx_bonded_H2 in idxs_bonded_H2:
                        candidates_hbonds.append([idx_type1, idx_bonded_H1, idx_mediated, idx_bonded_H2, idx_type2])

                # type2 is donor and type 1 is acceptor
                for idx_bonded_H3 in idxs_bonded_H3:
                    for idx_bonded_H2 in idxs_bonded_H2:
                        candidates_hbonds.append([idx_type2, idx_bonded_H3, idx_mediated, idx_bonded_H2, idx_type1])

            candidates_hbonds = np.array(candidates_hbonds)

            if candidates_hbonds.size == 0:
                print("No candidates found")
                info = np.zeros((18))
                info[0] = ts.frame
                info[1:] = np.nan
                all_info.append(info)
                continue

            # check if the h bond forms
            #
            # angle < OaOdHd <= 30



            poses_donor_o = u.atoms.positions[candidates_hbonds.T[0]]
            poses_bonded_H1 = u.atoms.positions[candidates_hbonds.T[1]]
            poses_mediated_o = u.atoms.positions[candidates_hbonds.T[2]]
            poses_bonded_H2 = u.atoms.positions[candidates_hbonds.T[3]]
            poses_acceptor_o = u.atoms.positions[candidates_hbonds.T[4]]

            angles_1 = calc_angles(poses_mediated_o, poses_donor_o, poses_bonded_H1, box=u.dimensions)/np.pi*180
            angles_2 = calc_angles(poses_acceptor_o, poses_mediated_o, poses_bonded_H2, box=u.dimensions)/np.pi*180
            mask = (angles_1 <= hbond_OaOdH_angle) & (angles_2 <= hbond_OaOdH_angle)

            hbonds = candidates_hbonds[mask]


            # update the poses
            poses_donor_o = u.atoms.positions[hbonds.T[0]]
            poses_bonded_H1 = u.atoms.positions[hbonds.T[1]]
            poses_mediated_o = u.atoms.positions[hbonds.T[2]]
            poses_bonded_H2 = u.atoms.positions[hbonds.T[3]]
            poses_acceptor_o = u.atoms.positions[hbonds.T[4]]



            d_DM, angles_MDH = get_hbond_dist_angle(poses_donor_o, poses_bonded_H1, poses_mediated_o, u.dimensions)
            d_MA, angles_AMH = get_hbond_dist_angle(poses_mediated_o, poses_bonded_H2, poses_acceptor_o, u.dimensions)
            # calculate delta
            d_DH1 = calc_bonds(poses_donor_o, poses_bonded_H1, box=u.dimensions)
            d_MH1 = calc_bonds(poses_mediated_o, poses_bonded_H1, box=u.dimensions)
            delta_DM = d_DH1 - d_MH1
            d_MH2 = calc_bonds(poses_mediated_o, poses_bonded_H2, box=u.dimensions)
            d_AH2 = calc_bonds(poses_acceptor_o, poses_bonded_H2, box=u.dimensions)
            delta_MA = d_MH2 - d_AH2

            d_DMA = (d_DM + d_MA)/2
            delta = (delta_DM + delta_MA)/2


            # information
            # [frame, donor_index, hydrogen1_index, mediated_index, hydrogen2_index, acceptor_index, d_DM, d_MA, angle_MDH, angle_AMH, d_DH1, d_MH1, d_MH2, d_AH2, delta_DM, delta_MA, (d_DM+d_MA)/2, (delta_DM+delta_MA)/2]
            if len(hbonds) == 0:
                info = np.zeros((18))
                info[0] = ts.frame
                info[1:] = np.nan
            else:
                info = np.zeros((len(hbonds), 18))
                info[:, 0] = ts.frame
                info[:, 1] = hbonds.T[0]
                info[:, 2] = hbonds.T[1]
                info[:, 3] = hbonds.T[2]
                info[:, 4] = hbonds.T[3]
                info[:, 5] = hbonds.T[4]
                info[:, 6] = d_DM
                info[:, 7] = d_MA
                info[:, 8] = angles_MDH
                info[:, 9] = angles_AMH
                info[:, 10] = d_DH1
                info[:, 11] = d_MH1
                info[:, 12] = d_MH2
                info[:, 13] = d_AH2
                info[:, 14] = delta_DM
                info[:, 15] = delta_MA
                info[:, 16] = d_DMA
                info[:, 17] = delta

                # pick the one with the smallest delta
                info = info[np.argmin(info, axis=0)[17]]

            # check the shape
            if info.shape[0] != 18:
                raise ValueError("The shape of info is not correct")
            #print(info)

            all_info.append(info)



    all_info = np.array(all_info)
    np.save("Surface-Indirect-PT-CV-Osd.npy", all_info, allow_pickle=False)
