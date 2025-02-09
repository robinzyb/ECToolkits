from pathlib import Path
import filecmp

import pytest

from ectoolkits.plots.dpmd import plot_dptest

import numpy as np

e_file = f"dptest.e.out"
f_file = f"dptest.f.out"
save_png = f"dptest.png"
rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz = \
    plot_dptest(e_file=e_file, f_file=f_file, save_name=save_png, return_err=True)

np.save("dptest_err.npy", [rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz])
# here I just simply compre the two files.