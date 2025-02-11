# Construct free energy surface through reweighting

## Introduction

TBC

## Quick start

```python
from ectoolkits.analysis.reweight import calc_FES, compute_bw_silverman

# compute bandwidths of CVs using Silverman's rule
colvar_file = Path(f"path_to_plumed_COLVAR_file")

T = 350 # temperature in Kelvin
kbT = T*0.0083144621 # to kJ/mol, which is a PLUMED internal unit

bias_factor = 44.67259179984373 # the bias factor in an OPES simulation, you can find it in the head of a STATE_WFILE file

bias = np.loadtxt(colvar_file, usecols=3) # usually, bias is in the third column of a colvar file.

ptcv = np.loadtxt(f"path_to_cv_files") # load CV data, which can be loaded from a PLUMED COLVAR file.
cv1 = ptcv[:,1]
cv2 = ptcv[:,2]
weights = np.exp(bias/kbT)
bw_cv1 = compute_bw_silverman(cv1, dim=2, weights=weights, bias_factor=bias_factor)
bw_cv2 = compute_bw_silverman(cv2, dim=2, weights=weights, bias_factor=bias_factor)
print(f"bandwidth for cv1 and cv2 are: {bw_cv1:0.6f}, {bw_cv2:0.6f}")


# compute fes
nbins_x = 100
nbins_y = 100
sigma_x = bw_cv1
sigma_y = bw_cv2
T = 350

# kbT and bias must have the same unit
kbT = T*0.0083144621 # kJ/mol
bias = np.loadtxt(colvar_file, usecols=3) # kJ/mol

ptcv = np.loadtxt(f"path_to_cv_files") # load CV data, which can be loaded from a PLUMED COLVAR file.
cv1 = ptcv[:,1]
cv2 = ptcv[:,2]

cv1_min = -1.5
cv1_max = 1.5
cv2_min = 2.3
cv2_max = 3.3

fes = calc_FES(cv_x=cv1,
               cv_y=cv2,
               bias=bias,
               bandwidth_x=bw_cv1,
               bandwidth_y=bw_cv2,
               nbins_x=nbins_x,
               nbins_y=nbins_y,
               kbT=kbT,
               save_file=True,
               name_file=f"PT/fes-surface-PT-pot{i}.dat",
               min_cv_x=cv1_min,
               max_cv_x=cv1_max,
               min_cv_y=cv2_min,
               max_cv_y=cv2_max,
               dim=2)

```
