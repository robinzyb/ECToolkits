# Introduction

A toolbox collect some postprocessing workflow



## quick band alignment 
```python
from toolkit.analysis.band_align import BandAlign
inp = {
     "input_type": "cube", 
     "ave_param":{
         "prefix": "./00.interface/hartree/Hematite-v_hartree-1_", 
         "index": (1, 502), 
         "l1": 4.8, 
         "l2": 4.8, 
         "ncov": 2, 
         "save": True,  
         "save_path":"00.interface"
     },
     "shift_param":{
         "surf1_idx": [124, 125, 126, 127, 128, 129, 130, 131],
         "surf2_idx": [24, 25, 26, 27, 28, 29, 30, 31]
     },
     "water_width_list": [8, 9, 9.5, 10, 10.5, 11, 12, 13],
     "solid_width_list": [1, 2, 3, 4]

}
ba = BandAlign(inp)
#quick view of hartree fluctuation
fig = ba.plot_hartree_per_width('water')
fig = ba.plot_hartree_per_width('solid')

# detail information is accessible in 
ba.water_hartree_list
ba.solid_hartree_list
```