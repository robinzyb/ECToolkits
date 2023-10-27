

```python
# Usage exmaple
cell = Paramcell(length=[19.63424, 19.237425, 68.35131], divi=[80, 75, 270], h=0)

# the position of a defect
pos = np.array([0.0, 0.0, 22.75066772733162423], dtype=float)*1.88972613288564
Q = 1.0
width= 1.0
recip = True
rho = GaussCharge(Q, pos, width, cell, recip=recip)


z_interface_list = np.array([13.303917-1.8, 43.534849+1.8])
diel_list = np.array([1.78, 4.75, 1.78])
beta_list = np.array([1.0, 1.0, 1.0])
diel_profile = DielProfile(z_interface_list, diel_list, beta_list, cell)

pbc =  PBCPoissonSolver(rho, diel_profile, cell)
```