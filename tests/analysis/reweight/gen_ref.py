from pathlib import Path
import numpy as np
from ectoolkits.analysis.reweight import calc_FES, calc_property_surface


case_fes_list = ["00.case_2d_fes"]
for case in case_fes_list:
    case = Path(case)
    ref_file = case/"fes_ref.dat"
    if ref_file.exists():
        print("Reference file exists, skip")
        continue
    colvar_file = case/"COLVAR"
    cv1, cv2, bias = np.loadtxt(colvar_file, usecols=(1, 2, 3), unpack=True)

    grid_bin_x = 10
    grid_bin_y = 10
    T = 350
    kbT = T*0.0083144621 # kJ/mol
    cv1_min = 2.3
    cv1_max = 2.4
    cv2_min = 1.0
    cv2_max = 2.0
    bw_cv1 = 0.008
    bw_cv2 = 0.008

    fes = calc_FES(cv_x=cv1,
                cv_y=cv2,
                bias=bias,
                bandwidth_x=bw_cv1,
                bandwidth_y=bw_cv2,
                nbins_x=grid_bin_x,
                nbins_y=grid_bin_y,
                kbT=kbT,
                save_file=True,
                name_file=ref_file,
                min_cv_x=cv1_min,
                max_cv_x=cv1_max,
                min_cv_y=cv2_min,
                max_cv_y=cv2_max,
                dim=2)
    

case_prop_list = ["01.case_2d_property"]
for case in case_prop_list:
    case = Path(case)
    ref_file = case/"prop_ref.dat"
    if ref_file.exists():
        print("Reference file exists, skip")
        continue
    colvar_file = case/"COLVAR"
    cv1, cv2, bias = np.loadtxt(colvar_file, usecols=(1, 2, 3), unpack=True)

    prop_data = np.load(case/"property_data.npy")
    grid_bin_x = 10
    grid_bin_y = 10
    T = 350
    kbT = T*0.0083144621 # kJ/mol

    cv1_min = 2.3
    cv1_max = 2.4
    cv2_min = 1.0
    cv2_max = 2.0
    bw_cv1 = 0.008
    bw_cv2 = 0.008

    fes = calc_property_surface(cv_x=cv1,
                                cv_y=cv2,
                                prop=prop_data,
                                bias=bias,
                                bandwidth_x=bw_cv1,
                                bandwidth_y=bw_cv2,
                                nbins_x=grid_bin_x,
                                nbins_y=grid_bin_y,
                                kbT=kbT,
                                save_file=True,
                                name_file=ref_file,
                                min_cv_x=cv1_min,
                                max_cv_x=cv1_max,
                                min_cv_y=cv2_min,
                                max_cv_y=cv2_max,
                                dim=2)

