# Acidity Calculation
We refer to paper for detailed description
>1. Cheng, J., Sulpizi, M. & Sprik, M. Redox potentials and pKa for benzoquinone from density functional theory based molecular dynamics. J Chem Phys 131, 154504 (2009).
>2. Mangold, M. et al. Absolute pK a Values and Solvation Structure of Amino Acids from Density Functional Based Molecular Dynamics Simulation. J Chem Theory Comput 7, 1951–1961 (2011).

The pKa of a species $\ce{AH}$ in aqueous solution is defined as
$$\ce{AH(aq) -> A^- (aq) + H^+ (aq)}$$
$$\mathrm{p}K_a = -\log{K_a}$$
The full expression for $\mathrm{p}K_a$ is
$$2.3 k_{\mathrm{B}} T\mathrm{p}K_a =   (\Delta_{\mathrm{dp}} A_{\ce{AH}} - \Delta_{\mathrm{dp}}  A_{\ce{H3O+}} - \Delta A_{\ce{Ad}} + \Delta A_{\ce{H2Od}} - \Delta A_{\mathrm{qc}} (\ce{AH}) + \Delta A_{\mathrm{qc}}(\ce{H3O+}) +  \Delta A_{\ce{H3O+}})$$

## Calculate Dummy Insertion Free Energy

$\Delta A_{\ce{Ad}}$ is dummy inerstion free energy corresponding to
$$\ce{A^-(aq) + H+(g)->Ad^-(aq) }$$
To calculate $\Delta A_{\ce{Ad}}$, you need calculate $\ce{Ad-}$ the vibrational frequency of mode i for dummy in gas phase. And save the frequncies as numpy array. Note that the unit of frequencies must be $cm^{-1}$. This function is straightforward implementation using eq.26 in reference[2].
```python
from ectoolkits.analysis.acidity import get_dummy_insert_fe
import numpy as np

# the frequencies you calculated from gas phase molecules
# these frequencies are taken from reference [2] for the arginine molecule.
frequencies_list = np.array([440, 1182, 2290])

get_dummy_insert_fe(frequencies_list, T=298)
```
The following output is
```shell
> 0.315428
```

We also implemented function for obtaining $\Delta A_{\ce{H2Od}}$, since $\Delta A_{\ce{H2Od}}$ has a special formula for correction, as described in reference[1]. To obtain the $\Delta A_{\ce{H2Od}}$, use the following code,
```python
from ectoolkits.analysis.acidity import get_dummy_insert_fe_hydronium

get_dummy_insert_fe_hydronium()
```
You will obtain `0.334` eV as the result, which is actually a constant.

## Calculate Quantum Correction Free Energy

$\Delta A_{\mathrm{qc}} (\ce{AH})$ is Nuclear Quantum Effects which are expected to be significant for proton. To calculate it, one need calculate vibrational frequencies for a gas phase molecule $\ce{AH}$. The units of frequencies must be $cm^{-1}$
```python
from ectoolkits.analysis.acidity import get_quantum_correction
import numpy as np
# the frequencies you calculated from gas phase molecules
# these frequencies are taken from reference [2] for the arginine molecule.
frequencies_list = np.array([263, 1204, 3340])
get_quantum_correction(frequencies_list, T=298)
```

The following output is
```shell
> 0.175
```

We also implemented function for obtaining $\Delta A_{\mathrm{qc}}(\ce{H3O+})$, since $\Delta A_{\mathrm{qc}}(\ce{H3O+})$ has a special formula for correction, as described in reference[2]. To obtain the $\Delta A_{\mathrm{qc}}(\ce{H3O+})$, use the following code,
```python
from ectoolkits.analysis.acidity import get_quantum_correction_hydronium

get_quantum_correction_hydronium()
```
You will obtain `0.192` eV as the result, which is actually a constant.
