# Workflow: Calculation of Dielectric Constant

**This workflow is under testing and not yet to be used**

**This workflow is powered by dpdispatcher**

## Usage
```shell
eckits wkflow calc_diel -i input.yaml -m machine.yaml -r resources.yaml
```

Example: `input.yaml`:
```yaml
input_file: "template.inp"                                                      
eps_type: "optical"                                                             
scale: "global"                                                                 
intensity_array: [0.0000, 0.0001, 0.0002]                                       
displacement_field: false                                                       
polarisation: [0.0, 0.0, 1.0]                                                   
d_filter: [0.0, 0.0, 1.0]                                                       
output_dir: "./output"                                                          
command: "srun cp2k.psmp -i input.inp"                                          
extra_forward_files: [                                                          
  "anatase_bulk_relaxed.xyz"                                                    
]                                                                               
extra_forward_common_files: [                                                   
]                                                                               
restart_wfn:   "anatase-RESTART.wfn"
```

Example: `machine.yaml`:
```yaml
batch_type: "Slurm"                                                             
context_type: "LocalContext"                                                    
local_root: "./"                                                                
remote_root: "./work_dir"
```

Example: `resources.yaml`:
```yaml
cpu_per_node: 12
gpu_per_node: 1
number_node: 4                                                                  
exclude_list: []                                                                
custom_flags:                                                                   
  - "#SBATCH --job-name='miniwkflow'"                                           
  - "#SBATCH --account='blabla'"                                                 
  - "#SBATCH --mail-type=ALL"                                                   
  - "#SBATCH --mail-user=blabla"                                
  - "#SBATCH --constraint=gpu"                                                  
  - "#SBATCH --cpus-per-task=1"                                                 
  - "#SBATCH --time=24:00:00"                                                   
source_list: []                                                                 
module_list:                                                                    
  - "daint-gpu"                                                                 
  - "CP2K"                                                                      
envs:                                                                           
  OMP_NUM_THREADS: "$SLURM_CPUS_PER_TASK"                                       
  CRAY_CUDA_MPS: "1"                                                            
prepend_script:                                                                 
  - "ulimit -s unlimited"                                                       
time_limit: "24:00:00"                                                          
queue_name: "normal"                                                            
group_size: 1       
```

## Principle
ECToolkits provides the workflow for calculating ab initio dielectric constant using CP2K.

For detail description, users are referred to this paper {cite}`Umari.2002.10.1103/physrevlett.89.157602`.


Basically, we need to calculate dipole moments with varying electric fields. Then, fit the data to the following equation:

$$
\varepsilon = 4\pi\frac{M}{\Omega E} + 1 
$$




# Bibliography
```{bibliography}
```