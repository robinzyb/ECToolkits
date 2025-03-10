# Calculate collective variables of proton transfer reactions

## Introduction

References (angelos JPCL, Quaranta JPCL, Marcos Chem. Sci., Jia Precise Chemistry)

## Usage

The `ProtonTransferCV` class is able to compute the collective variables for proton transfer reactions, that is, $\delta$ and $d$.
1. load a MDAnalysis `Universe` class from your trajectory files. It also requires the information of chemical symbols. Add the chemical symbols to topology attributes as `types`.
2. provide the indices (0-based) of atoms belonging to two different groups, `idxs_type1_o`, `idxs_type2_o`. The atoms can either be hydrogen donor or be hydrogen acceptor during proton transfer reactions. The groups are customized by users depending on specific contexts.
3. provide the number of bridge water molecules `num_bridge`. In some cases, proton transfer may involve other water molecules. These molecules are called bridge molecules. The mechanism of this type of proton transfer is called `Grotthuss Mechanism`. You can defined the `num_bridge=1` to calculate collective variables of `Grotthuss`-type proton transfer. By default, `num_bridge` is `0`. For the moment being, `num_bridge > 1` is not supported. Thus, the `num_bridge` should either be `0` or `1`.
4. provide the indices (0-based) of oxygen atoms in potential bridge water molecules. If you set `num_bridge = 1`, you should provide the indices of potential bridge water molecules. The indices should not be overlaped with that in `idxs_type1_o` and `idxs_type2_o`.


### Direct proton transfer

```python
import numpy as np
from MDAnalysis import Universe
from ase.io import read

from ectoolkits.analysis.proton_transfer_cv import ProtonTransferCV

# load Universe from lammps-dump-text (lammps trajectory)
lammpstrj = "clip.lammpstrj"
stc = read(lammpstrj, format="lammps-dump-text", specorder=["Bi", "H", "O", "V"])
chemical_symbols = stc.get_chemical_symbols()
u= Universe(lammpstrj, format="LAMMPSDUMP")
u.add_TopologyAttr("types", np.array(chemical_symbols))


atomgroup = u.atoms
# donor/acceptor of type 1 atoms
idxs_type1_o = [150]
# donor/acceptor of type 2 atoms
idxs_type2_o = [130, 241, 136, 139, 129]

ptcv = ProtonTransferCV(atomgroup=atomgroup, idxs_type1_o=idxs_type1_o, idxs_type2_o=idxs_type2_o, extra_detail=True)

ptcv.run()

```
The results of the analysis can be accessed through the `results` or `df` attributes. `ProtonTransferCV.results` provides results in the Numpy Array format while `ProtonTransferCV.df` in the DataFrames format.
```python
# in Numpy Array
ptcv.results
# in DataFrame
ptcv.df
```
|    |   Frame |   Delta_CV |   Distance_CV |   Index of donor0 |   Index of hydrogen0 |   Index of acceptor |   Distance (Donor0-Hydrogen0) |   Distance (Acceptor0-Hydrogen0) |   Interatomic distance (Donor0-Aceeptor0) |   Angle (Donor0-Aceeptor0-Hydrogen0) |
|---:|--------:|-----------:|--------------:|------------------:|---------------------:|--------------------:|------------------------------:|---------------------------------:|------------------------------------------:|-------------------------------------:|
|  0 |       0 |  -0.814423 |       2.66147 |               150 |                  153 |                 130 |                      0.97808  |                          1.7925  |                                   2.66147 |                             22.0054  |
|  1 |       1 |  -0.677237 |       2.65874 |               150 |                  153 |                 130 |                      1.00409  |                          1.68133 |                                   2.65874 |                             10.4766  |
|  2 |       2 |  -0.682127 |       2.63129 |               150 |                  153 |                 130 |                      1.0112   |                          1.69333 |                                   2.63129 |                             17.37    |
|  3 |       3 |  -0.918069 |       2.71045 |               150 |                  153 |                 130 |                      0.970081 |                          1.88815 |                                   2.71045 |                             26.0992  |
|  4 |       4 | nan        |     nan       |               nan |                  nan |                 nan |                    nan        |                        nan       |                                 nan       |                            nan       |
|  5 |       5 |   0.255239 |       2.4176  |               136 |                  293 |                 150 |                      1.08379  |                          1.33903 |                                   2.4176  |                              4.17919 |

if the `extra_detail` keyword is set to `False`, only the frame index and collective variables are shown.
```python
ptcv = ProtonTransferCV(atomgroup=atomgroup, idxs_type1_o=idxs_type1_o, idxs_type2_o=idxs_type2_o, extra_detail=False)

ptcv.run()
ptcv.df
```
|    |   Frame |   Delta_CV |   Distance_CV |
|---:|--------:|-----------:|--------------:|
|  0 |       0 |  -0.814423 |       2.66147 |
|  1 |       1 |  -0.677237 |       2.65874 |
|  2 |       2 |  -0.682127 |       2.63129 |
|  3 |       3 |  -0.918069 |       2.71045 |
|  4 |       4 | nan        |     nan       |
|  5 |       5 |   0.255239 |       2.4176  |
### Indirect proton transfer

The analysis for indirect proton transfer is similar to direct proton transfer. In addition to the parameters, you have to set `num_bridge=1` and provide the indices of oxygen atoms of potential bridge water molecules. The following is an example.

```python

atomgroup = u.atoms
idxs_type1_o = [150]
idxs_type2_o = [130, 241, 136, 139, 129]
idxs_water = [144, 151, 157, 162, 168, 169, 180, 183, 185, 186, 187, 188, 189, 191, 216, 217, 222, 223, 224, 226, 227, 228, 229, 232, 233, 234, 235, 236, 239, 240, 242, 243, 244, 247, 315, 321, 324, 327, 330, 339, 342, 345, 348, 351, 354, 130, 145, 156, 163, 174, 175, 181, 182, 184, 190, 218, 219, 220, 221, 225, 230, 231, 237, 238, 245, 246, 312, 318, 333, 336, 357]

ptcv = ProtonTransferCV(atomgroup=atomgroup, idxs_type1_o=idxs_type1_o, idxs_type2_o=idxs_type2_o, num_bridge=1, idxs_water_o=idxs_water, extra_detail=True)
ptcv.run()

```

|    |   Frame |   Delta_CV |   Distance_CV |   Index of donor0 |   Index of hydrogen0 |   Index of donor1 |   Index of hydrogen1 |   Index of acceptor |   Distance (Donor0-Hydrogen0) |   Distance (Donor1-Hydrogen1) |   Distance (Acceptor0-Hydrogen0) |   Distance (Acceptor1-Hydrogen1) |   Interatomic distance (Donor0-Aceeptor0) |   Interatomic distance (Donor1-Aceeptor1) |   Angle (Donor0-Aceeptor0-Hydrogen0) |   Angle (Donor1-Aceeptor1-Hydrogen1) |
|---:|--------:|-----------:|--------------:|------------------:|---------------------:|------------------:|---------------------:|--------------------:|------------------------------:|------------------------------:|---------------------------------:|---------------------------------:|------------------------------------------:|------------------------------------------:|-------------------------------------:|-------------------------------------:|
|  0 |       0 | nan        |     nan       |               nan |                  nan |               nan |                  nan |                 nan |                    nan        |                      nan      |                        nan       |                        nan       |                                 nan       |                                 nan       |                             nan      |                            nan       |
|  1 |       1 |  -0.923249 |       2.84656 |               150 |                  257 |               233 |                  252 |                 129 |                      0.931612 |                        1.0139 |                          2.03899 |                          1.75302 |                                   2.94252 |                                   2.75059 |                              11.6913 |                              8.19724 |

