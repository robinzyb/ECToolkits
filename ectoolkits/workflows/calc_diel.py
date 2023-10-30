import numpy.typing as npt
import numpy as np
from typing import Dict
from typing import List
from pathlib import Path
import shutil
from dpdispatcher import Machine, Resources, Task, Submission
from cp2k_input_tools.parser import CP2KInputParser, CP2KInputParserSimplified
from cp2k_input_tools.generator import CP2KInputGenerator

DIPOLE_MOMENT_FILE = "moments.dat"
def copy_file_list(file_list, target_dir):
    target_dir = Path(target_dir)
    for file in file_list:
        file = Path(file)
        file_basename = file.name
        src = file
        dst = target_dir/file_basename
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
            print(f"COPY directory {src}")
            print(f"TO {dst}")
        elif src.is_file():
            shutil.copy2(src, dst)
            print(f"COPY file {src}")
            print(f"TO {dst}")

def gen_cp2k_input_dict(input_file: str, 
                        canonical: bool
                        ):
    if canonical:
        parser = CP2KInputParser()
    else:
        parser = CP2KInputParserSimplified()
    with open(input_file) as fhandle:
        input_dict = parser.parse(fhandle)
    return input_dict

def write_cp2k_input(input_dict: Dict, 
                     file_name: str
                     ):
    generator = CP2KInputGenerator()
    with open(file_name, "w") as fhandle:
        for line in generator.line_iter(input_dict):
            fhandle.write(f"{line}\n")

def add_efield_input(input_dict: Dict, 
                     intensity: float,
                     displacement_field: bool,
                     polarisation: npt.NDArray[np.float64],
                     d_filter: npt.NDArray[np.float64], 
                     ):
    
    # Add the efield input to the input dictionary
    assert len(input_dict['+force_eval']) == 1, \
        "Only one FORCE_EVAL is supported for now"
    input_dict['+force_eval'][0]['+dft']['+periodic_efield'] = {}
    input_dict['+force_eval'][0]['+dft']['+periodic_efield']['intensity'] = intensity
    input_dict['+force_eval'][0]['+dft']['+periodic_efield']['displacement_field'] = displacement_field
    input_dict['+force_eval'][0]['+dft']['+periodic_efield']['polarisation'] = polarisation
    input_dict['+force_eval'][0]['+dft']['+periodic_efield']['d_filter'] = d_filter

    return input_dict

def add_print_moments(input_dict: Dict,
                      periodic: bool,
                      filename: str,
                      ):
    # Add the print moments input to the input dictionary
    assert len(input_dict['+force_eval']) == 1, \
        "Only one FORCE_EVAL is supported for now"
    input_dict['+force_eval'][0]['+dft']['+print'] = {
        '+moments': {}
    }
    input_dict['+force_eval'][0]['+dft']['+print']['+moments']['periodic'] = periodic
    input_dict['+force_eval'][0]['+dft']['+print']['+moments']['filename'] = filename
    return input_dict

def add_run_type(input_dict: Dict,
                 run_type: str,
                 ):
    # Add the run type input to the input dictionary
    assert len(input_dict['+force_eval']) == 1, \
        "Only one FORCE_EVAL is supported for now"
    input_dict['+global']['run_type'] = run_type
    return input_dict

def add_restart_wfn(input_dict: Dict,
                    restart_wfn: str,
                    ):
    # Add the restart wfn path to the input dictionary
    assert len(input_dict['+force_eval']) == 1, \
        "Only one FORCE_EVAL is supported for now"
    restart_wfn = Path(restart_wfn)
    # always make sure the wfn is only one level higher than the input file
    input_dict['+force_eval'][0]['+dft']['wfn_restart_file_name'] = \
        f"../{restart_wfn.name}"
    return input_dict

def gen_series_calc_efield(input_dict: Dict,
                           intensity_array: npt.NDArray[np.float64],
                           displacement: bool,
                           polarisation: npt.NDArray[np.float64],
                           d_filter: npt.NDArray[np.float64],
                           periodic: bool,
                           eps_type: str,
                           filename: str,
                           output_dir: str,
                           extra_forward_files: List[str],
                           restart_wfn: str=None,
                           ):
    
    # store the path for each calculation
    task_work_path_list = []
    # produce input files for each calculation
    output_dir = Path(output_dir)

    # Add the print moments input to the input dictionary
    input_dict = add_print_moments(input_dict, periodic, filename)
    if eps_type == "optical":
        input_dict = add_run_type(input_dict, "ENERGY_FORCE")
    elif eps_type == "static":
        input_dict = add_run_type(input_dict, "GEO_OPT")


    if restart_wfn is not None:
        input_dict = add_restart_wfn(input_dict, restart_wfn)
    # Add the efield input to the input dictionary
    for intensity in intensity_array:
        input_dict = add_efield_input(input_dict, intensity, displacement, polarisation, d_filter)

        # Write the input dictionary to a file
        single_calc_dir = output_dir/f"efield_{intensity:7.6f}"
        single_calc_dir.mkdir(parents=True, exist_ok=True)
        # task_work_path should be relative to the work_base
        task_work_path_list.append(single_calc_dir.name)

        output_file = single_calc_dir/"input.inp"
        write_cp2k_input(input_dict, output_file)
        print(f"Input file for efield {intensity:7.6f} written to {output_file}")
        copy_file_list(extra_forward_files, single_calc_dir)

    return task_work_path_list

def gen_task_list(command, task_work_path_list, extra_forward_files):
    # generate task list
    task_list = []

    outlog = "cp2k.log"
    forward_files = extra_forward_files + ["input.inp"]
    backward_files = [DIPOLE_MOMENT_FILE, outlog]
    for task_work_path in task_work_path_list:
        task = Task(command=command, 
                    task_work_path=task_work_path,
                    forward_files=forward_files,
                    backward_files=backward_files,
                    outlog=outlog)
        task_list.append(task)
    return task_list

def calc_diel(input_file: str,
              intensity_array: npt.NDArray[np.float64],
              displacement_field: bool,
              polarisation: npt.NDArray[np.float64],
              d_filter: npt.NDArray[np.float64],
              eps_type: str,
              output_dir: str,
              machine_dict: Dict,
              resources_dict: Dict,
              command: str,
              extra_forward_files: List[str]=[],
              extra_forward_common_files: List[str]=[],
              restart_wfn: str=None,
              ):
    # gen input dict
    template_input_dict = gen_cp2k_input_dict(input_file, canonical=True)
    # gen task work path list
    task_work_path_list = gen_series_calc_efield(template_input_dict, 
                                                 intensity_array, 
                                                 displacement_field, 
                                                 polarisation, 
                                                 d_filter, 
                                                 periodic=True, 
                                                 eps_type=eps_type,
                                                 filename="="+DIPOLE_MOMENT_FILE, 
                                                 output_dir=output_dir,
                                                 extra_forward_files=extra_forward_files,
                                                 restart_wfn=restart_wfn
                                                 )
    print(task_work_path_list)
    # gen task
    task_list = gen_task_list(command, task_work_path_list, extra_forward_files)
    # submission
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    # to absolute path
    #TODO: bug here the common files cannot be uploaded using LazyLocalContext.
    forward_common_files = extra_forward_common_files
    if restart_wfn:
        forward_common_files.append(restart_wfn)
    # copy to the work base directory so that it can be uploaded
    copy_file_list(forward_common_files, output_dir)
    # workbase will be transfer to absolute path
    # local_root/taskpath is the full path for upload files 
    submission = Submission(work_base=output_dir,
                            machine=machine,
                            resources=resources, 
                            task_list=task_list, 
                            forward_common_files=forward_common_files,
                            backward_common_files=[],
                            )
    submission.run_submission()
    
    print("Workflow for Calculation of Dielectric Constant Complete!")