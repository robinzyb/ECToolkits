from cp2kdata.cube import Cp2kCube
import numpy as np
import os
import matplotlib.pyplot as plt


def get_pav_list(prefix, index, save=True, axis='z', save_path="."):
    # index last number is exclude
    x_list = []
    pav_list = []
    for idx in range(*index):
        cube = Cp2kCube(f"{prefix}{idx}.cube")
        x, pav = cube.get_pav()
        x_list.append(x)
        pav_list.append(pav)
        print(f"process cube {idx} finished", end="\r")
    x_list = np.array(x_list)
    pav_list = np.array(pav_list)
    if save:
        np.savetxt(os.path.join(save_path, "x_list.dat"), x_list, fmt="%3.4f")
        np.savetxt(os.path.join(save_path, "pav_list.dat"), pav_list, fmt="%3.4f")
    return x_list, pav_list

def get_mav_list(prefix, index, l1, l2=0, ncov=2, save=True, axis='z', save_path="."):
    # index last number is exclude
    x_list = []
    mav_list = []
    for idx in range(*index):
        cube = Cp2kCube(f"{prefix}{idx}.cube")
        x, mav = cube.get_mav(l1=l1, l2=l2, ncov=ncov, interpolate=True)
        x_list.append(x)
        mav_list.append(mav)
        print(f"process cube {idx} finished", end="\r")
    x_list = np.array(x_list)
    mav_list = np.array(mav_list)
    if save:
        np.savetxt(os.path.join(save_path, "x_list.dat"), x_list, fmt="%3.4f")
        np.savetxt(os.path.join(save_path, "mav_list.dat"), mav_list, fmt="%3.4f")
    return x_list, mav_list

def get_nearest_idx(array, value):
    idx = np.argmin(np.abs(array-value))
    return idx

def get_slab_cent(traj, surf1_idx, surf2_idx, cell_z):
    slab_cent_list = []
    for snapshot in traj:
        surf1_z = get_z_mean(snapshot, surf1_idx)
        surf2_z = get_z_mean(snapshot, surf2_idx)
        if surf1_z < surf2_z:
            surf2_z =- cell_z
        slab_cent = (surf1_z + surf2_z)/2
        slab_cent_list.append(slab_cent)
    slab_cent_list = np.array(slab_cent_list)
    return slab_cent_list

def align_to_slab_cent(x_list, pav_list, traj, surf1_idx, surf2_idx, cell_z):
    # surf1 is solid on the left
    # surf2 is solid on the right
    # ok for pav_list and mav_list
    slab_cent_list = get_slab_cent(traj, surf1_idx, surf2_idx, cell_z)

    slab_cent_1st = slab_cent_list[0]
    slab_cent_1st_idx = get_nearest_idx(x_list[0], slab_cent_1st)
    new_pav_list = []

    for slab_cent, x, pav in zip(slab_cent_list, x_list, pav_list):
        slab_cent_idx = get_nearest_idx(x, slab_cent)
        roll_num = slab_cent_idx - slab_cent_1st_idx
        new_pav = np.roll(pav, -roll_num)
        new_pav_list.append(new_pav)
    new_pav_list = np.array(new_pav_list)
    return new_pav_list
