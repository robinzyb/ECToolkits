# Dissociation Degree Analysis
Dissociation degree affects the interfacial dipoles, hence, the band alignments of semiconductor-electrolyte interfaces. To calculate the dissociation degree for a given trajectory, we implemented the classes `CNState` and `DisDeg`, which are briefly introduced below.



# The CNState Class

## Analysis
The necessity of dissociation degree is the coordination number of atoms. For example, the dissociation degree of interfacial water molecules is the ratio of the number of adsorbed water molecules to the dissociated water molecules(usually the adsorbed OH groups). Therefore, one needs to count the coordination number of the adsorbed oxygen atoms for identifying the OH groups, then calculates the dissociation degrees. To this end, the `CNState` class calculates the coordination number of a list of atoms for a given trajectory.
```python
# First, read a trajectory using Universe. Do not forget add dimensions.
>>> from MDAnalysis import Universe
>>> xyzfile = "./tio2-water.xyz"
>>> cellpar = np.array([12.4, 11.238, 38.42, 90, 90, 90])
>>> u = Universe(xyzfile)
>>> u.dimensions = cellpar
# Second, use CNState class to analyze the trajectory
>>> from ectoolkits.analysis.disdeg import CNState
# O_idx_dw is the indexes of adsorbed oxygen atoms. You may obtained by yourself.
>>> O_idx_dw = [264, 267, 270, 273, 276, 279, 282, 285]
>>> cnstate = CNState(atomgroup=u.atoms,
>>>          center_atom_idx=np.array(O_idx_dw),
>>>          coordinated_elements=['H'],
>>>          max_cutoff=1.2)
>>> cnstate.run()
>>> cnstate._cnstate
array([[2, 2, 2, 2, 2, 2, 1, 2],
[2, 2, 2, 2, 2, 2, 1, 2],
[2, 2, 2, 2, 2, 2, 1, 2],
[2, 2, 2, 2, 2, 2, 1, 2],
[2, 2, 2, 2, 2, 2, 1, 2]])

```
The result is an array with shape of (num_frames, num_center_atom_idx). Each row of the array is the coordination numbers of center_atoms with respect to Hydrogen atoms. If one would like to calculate coordination numbers with respect to other elements, such as, "C", "N", one can set `coordinated_elements` as `["C", "N"]` etc.

## Plot CNStates
After finishing the analysis, the method `cnstate.plot(cn_list=[0, 1, 2, 3])` could quickly plot the percentages of all coordination number states.

## Save CNStates
The results are saved to npy files using the method `cnstate.save_cnstate(filename='cn.npy')`

## Restore the Analysis
`CNState` Class could also be restored from the `cn.npy` file using the method `CNState.read_cnstate_from(npyfile='cn.npy')`

# The DisDeg Class
The `DisDeg` class adds extra features to calculate dissociation degree, such as, `get_disdeg`,`save_disdeg`, and `plot_disdeg`.
The `DisDeg` is inherited from the `CNState` class, which means that the initialization is the same as `CNState`.

## Analysis
```python
# First, read a trajectory using Universe. Do not forget add dimensions.
>>> from MDAnalysis import Universe
>>> xyzfile = "./tio2-water.xyz"
>>> cellpar = np.array([12.4, 11.238, 38.42, 90, 90, 90])
>>> u = Universe(xyzfile)
>>> u.dimensions = cellpar
# Second, use CNState class to analyze the trajectory
>>> from ectoolkits.analysis.disdeg import DisDeg
# O_idx_dw is the indexes of adsorbed oxygen atoms. You may obtained by yourself.
>>> O_idx_dw = [264, 267, 270, 273, 276, 279, 282, 285]
>>> disdeg = DisDeg(atomgroup=u.atoms,
>>>          center_atom_idx=np.array(O_idx_dw),
>>>          coordinated_elements=['H'],
>>>          max_cutoff=1.2)
>>> disdeg.run()
# Third, obtain the dissociation degree
# Assign the coordination numbers of 2 and 3 to the intact water molecules and the coordination numbers of 0 and 1 to the dissociated water molecules.
# one can customize the cn_list_no_dis and cn_list_dis
cn_list_no_dis= [2, 3]
cn_list_dis= [0, 1]
>>> disdeg.get_disdeg(cn_list_no_dis=cn_list_no_dis, cn_list_dis=cn_list_dis)
```

## Plot DisDeg
After finishing the analysis, the method `disdeg.plot_disdeg()` could quickly plot the percentages of non-dissociated and dissociated degree.

## Save DisDeg
The results are saved to npy files using the method `disdeg.save_disdeg(filename='disdeg.npy')`
