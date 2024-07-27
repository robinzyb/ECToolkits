# Atom density analysis

Density analysis along the direction normal to an interface is very helpful for understanding the interface structure.{cite}`Fan.2023.10.1016/j.jechem.2023.03.013,Le.2017.10.1103/physrevlett.119.016801,Wen.2023.10.1021/acs.jpcc.3c00220,Guo.2018.10.1021/acs.chemmater.7b02619,Andrade.2020.10.1039/c9sc05116c` We can obtain the density by giving the following parameters as written in the `inp_dict`.

First of all, we need to define where the interfaces are using the surface atom indices.
These indices can be easily retreived using the class `Slab`.
Note that the `Slab` class is inherent from `Atoms` class in `ase`.
In this way, we can either use the methods built in the `Atoms` class or the new methods created in `ECToolkits`

To initialize the `Slab` class, we can simply use `Slab` class in the `Atoms` object.

```python
from ectoolkits.structures.slab import Slab
stc = Slab(stc)
```
There is a method called `find_idx_from_range` in `Slab` class.
If the z coordinates of surface atoms are located in the range from 6.2 Angstrom to 7.2 Angstrom,
we can obtain the indices using the information,
```python
surf1_idx = stc.find_idx_from_range(zmin=6.2, zmax=7.2, element='Sn')
```
Here, the `zmin` and `zmax` define the lower and upper bounds of z coordinates.
The spcification of `element` is optional. If we don't define the element, the indices of all atoms located within this range will be retrevied. Note that, currently, this method only supports to get the indices according to the z coordinates.

Now we need to define the `surf1` and `surf2`.
1. `surf1` is the interface on the left, where the solid part is in the left of liquid.
2. `surf2` is the interface on the right, where the solid part is in the right of liquid.

The reason two interfaces exist is that the model is symmetric with respect to the center plane of slab, as shown in the following figure.

![density](./figures/hartree_area.png)

Next, we move to the analysis of water density at interfaces.

To perform the water density analysis, the indices of oxygen atoms located between two interfaces are need to be specified. Again, we can find the indices using the method `find_idx_from_range`. And put these indices (`List`) in the `inp_dict["density_type"]["idx_list"]`. Here, the `density_unit` should be set to `"water"`, because we treat the coordinates of oxygen atoms as the position of water molecules and covert them to the water density by unit conversion.


Now, we import the analysis method `AtomDensity` and input the following mentioned parameters.
```python
from ectoolkits.analysis.atom_density import AtomDensity

# from
inp_dict={
     "xyz_file": "./Hematite-pos-1.xyz", # the path to the xyz trajecotry.
     "cell": [10.0564, 8.7091, 38.506, 90, 90, 90], # the cell parameters
     "surf2": [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], # the interface on the right
     "surf1": [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 12], # the interface on the left
     "density_type":[
         {
             "element": "O",
             "idx_method": "manual",
             "idx_list": O_idx,
             "density_unit": "water",
             "dz": 0.05,
             "name": "O_density"
             },
         {
             "element": "H",
             "idx_method": "manual",
             "idx_list": H_idx,
             "density_unit": "water",
             "dz": 0.05,
             "name": "H_density"
             }
         ]
 }

ad = AtomDensity(inp_dict)
ad.run()

# detail information is accessible in
ad.atom_density
ad.atom_density_z

# get average denstiy from center
width_list = [5, 6, 7, 8, 9, 10]
all_cent_density = ad.get_ave_density(width_list)

# quick plot for denstiy
# if you want to symmetrize the density profile, set sym=True
ad.plot_density(sym=False)

```
![density](./figures/density.png)