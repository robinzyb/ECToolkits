import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import Dict
from typing import List
from pathlib import Path
import shutil
from dpdispatcher import Machine, Resources, Task, Submission
from cp2k_input_tools.parser import CP2KInputParser, CP2KInputParserSimplified
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2kdata.block_parser.dipole import parse_dipole_list
from cp2kdata import Cp2kOutput
from cp2kdata import Cp2kCube
from cp2kdata.units import au2A
from ectoolkits.analysis.dielectric_constant import get_dielectric_constant_profile

DIPOLE_MOMENT_FILE = "moments.dat"
DENSITY_FILE = "*-ELECTRON_DENSITY*.cube"
CP2K_LOG_FILE = "cp2k.log"
debye2au = 4.26133088E-01/1.08312217E+00


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


def file_to_list(fname: str):
    with open(fname, 'r') as fp:
        output_file = fp.read()
    return output_file


def get_dipole_moment_array(task_work_path_list: List[str],
                            output_dir: str,
                            axis: str):
    index_dict = {
        'x': 0,
        'y': 1,
        'z': 2,
    }

    dipole_moment_array = []
    for task_work_path in task_work_path_list:
        output_dir = Path(output_dir)
        output_file = file_to_list(
            output_dir/task_work_path/DIPOLE_MOMENT_FILE)
        dipole_moment_array.append(parse_dipole_list(
            output_file)[0][index_dict[axis]])
    return np.array(dipole_moment_array)*debye2au


def get_volume_array(task_work_path_list: List[str],
                     output_dir: str,):
    volume_array = []
    for task_work_path in task_work_path_list:
        output_dir = Path(output_dir)
        cp2k_output = Cp2kOutput(output_dir/task_work_path/CP2K_LOG_FILE)
        cell = cp2k_output.get_all_cells()[0]
        volume = np.linalg.det(cell)/(au2A**3)
        volume_array.append(volume)
    return np.array(volume_array)


def get_dielectric_constant(dipole_moment_array: npt.NDArray[np.float64],
                            intensity_array: npt.NDArray[np.float64],
                            volume_array: npt.NDArray[np.float64],):

    polarization_array = dipole_moment_array/volume_array
    slope, intercept, r, p, se = linregress(
        intensity_array, polarization_array)
    dielectric_constant = slope * 4 * np.pi + 1
    return dielectric_constant


def plot_dielectric_fitting(intensity_array: npt.NDArray[np.float64],
                            dipole_moment_array: npt.NDArray[np.float64],
                            dielectric_constant: npt.NDArray[np.float64],
                            output_dir: str,
                            axis: str,
                            ):

    # for plotting
    slope, intercept, r, p, se = linregress(
        intensity_array, dipole_moment_array)

    output_dir = Path(output_dir)
    plt.style.use('cp2kdata.matplotlibstyle.jcp')
    row = 1
    col = 1
    fig = plt.figure(figsize=(3.37*col, 1.89*row), dpi=600, facecolor='white')
    gs = fig.add_gridspec(row, col)
    ax = fig.add_subplot(gs[0])

    ax.scatter(intensity_array, dipole_moment_array)

    xlim = ax.get_xlim()
    x = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(x, slope*x+intercept, "--")

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_ylabel("Dipole Moment [a.u.]")
    ax.set_xlabel(f"Efield {axis} [a.u.]")
    fig.savefig(output_dir/"dielectric_fitting.png",
                dpi=600, bbox_inches='tight')


def get_dielectic_constant_atomic(task_work_path_list: List[str],
                                  output_dir: str,
                                  intensity: npt.NDArray[np.float64],
                                  axis: str,
                                  ):
    output_dir = Path(output_dir)
    den_file_1 = list(
        (output_dir/task_work_path_list[0]).glob(DENSITY_FILE))[0]
    den_file_2 = list(
        (output_dir/task_work_path_list[1]).glob(DENSITY_FILE))[0]
    rho_cube_1 = Cp2kCube(den_file_1)
    rho_cube_2 = Cp2kCube(den_file_2)

    dipole_moment_array = get_dipole_moment_array(
        task_work_path_list, output_dir, axis)
    volume_array = get_volume_array(task_work_path_list, output_dir)
    polarization_array = dipole_moment_array/volume_array
    Delta_macro_polarization = polarization_array[1]-polarization_array[0]

    z, dielectric_constant = get_dielectric_constant_profile(rho_1=rho_cube_1,
                                                             rho_2=rho_cube_2,
                                                             Delta_macro_Efield=2.0*intensity,
                                                             Delta_macro_polarization=Delta_macro_polarization,
                                                             axis=axis)
    return z, dielectric_constant


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

    if '+print' not in input_dict['+force_eval'][0]['+dft']:
        input_dict['+force_eval'][0]['+dft']['+print'] = {}

    moment_dict = {
        '+moments': {
            'periodic': periodic,
            'filename': filename,
        }
    }
    input_dict['+force_eval'][0]['+dft']['+print'].update(moment_dict)
    return input_dict


def add_print_density(input_dict: Dict,
                      ):
    # Add the print moments input to the input dictionary
    assert len(input_dict['+force_eval']) == 1, \
        "Only one FORCE_EVAL is supported for now"

    if '+print' not in input_dict['+force_eval'][0]['+dft']:
        input_dict['+force_eval'][0]['+dft']['+print'] = {}

    density_dict = {
        '+e_density_cube': {
            '+each': {
                'geo_opt': '0'
            }
        }
    }
    input_dict['+force_eval'][0]['+dft']['+print'].update(density_dict)

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


def plot_dielectric_profile(z: npt.NDArray[np.float64],
                            dielectric_constant: npt.NDArray[np.float64],
                            output_dir: str,
                            axis: str,
                            ):
    output_dir = Path(output_dir)
    plt.style.use('cp2kdata.matplotlibstyle.jcp')
    row = 1
    col = 1
    fig = plt.figure(figsize=(3.37*col, 1.89*row), dpi=600, facecolor='white')
    gs = fig.add_gridspec(row, col)
    ax = fig.add_subplot(gs[0])
    ax.plot(z, dielectric_constant)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"$\varepsilon_{\infty}$")
    ax.set_xlabel(f"{axis} [Bohr]")
    fig.savefig(output_dir/"dielectric_profile.png",
                dpi=600, bbox_inches='tight')


def gen_calc_opposite_efield(input_dict: Dict,
                             intensity: npt.NDArray[np.float64],
                             displacement_field: bool,
                             axis: str,
                             periodic: bool,
                             eps_type: str,
                             filename: str,
                             output_dir: str,
                             extra_forward_files: List[str],
                             restart_wfn: str = None,
                             ):
    # for the atomic dielectric constant calculation
    direction_vector = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }
    # store the path for each calculation
    task_work_path_list = []
    # produce input files for each calculation
    output_dir = Path(output_dir)

    # Add the print moments input to the input dictionary
    input_dict = add_print_moments(input_dict, periodic, filename)
    input_dict = add_print_density(input_dict)
    if eps_type == "optical":
        input_dict = add_run_type(input_dict, "ENERGY_FORCE")
    elif eps_type == "static":
        input_dict = add_run_type(input_dict, "GEO_OPT")

    if restart_wfn is not None:
        input_dict = add_restart_wfn(input_dict, restart_wfn)

    # Add the efield input to the input dictionary

    for sign in [-1, 1]:
        input_dict = add_efield_input(input_dict=input_dict,
                                      intensity=intensity,
                                      displacement_field=displacement_field,
                                      polarisation=list(
                                          sign*direction_vector[axis]),
                                      d_filter=list(direction_vector[axis]))

        # Write the input dictionary to a file
        single_calc_dir = output_dir/f"efield_{(sign*intensity):7.6f}"
        single_calc_dir.mkdir(parents=True, exist_ok=True)
        # task_work_path should be relative to the work_base, i.e. output_dir
        task_work_path_list.append(single_calc_dir.name)

        output_file = single_calc_dir/"input.inp"
        write_cp2k_input(input_dict, output_file)
        print(
            f"Input file for efield {intensity:7.6f} written to {output_file}")
        copy_file_list(extra_forward_files, single_calc_dir)

    return task_work_path_list


def gen_series_calc_efield(input_dict: Dict,
                           intensity_array: npt.NDArray[np.float64],
                           displacement_field: bool,
                           axis: str,
                           periodic: bool,
                           eps_type: str,
                           filename: str,
                           output_dir: str,
                           extra_forward_files: List[str],
                           restart_wfn: str = None,
                           ):
    # for the global dielectric constant calculation
    direction_vector = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }
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
        input_dict = add_efield_input(input_dict=input_dict,
                                      intensity=intensity,
                                      displacement_field=displacement_field,
                                      polarisation=list(
                                          direction_vector[axis]),
                                      d_filter=list(direction_vector[axis]),
                                      )

        # Write the input dictionary to a file
        single_calc_dir = output_dir/f"efield_{intensity:7.6f}"
        single_calc_dir.mkdir(parents=True, exist_ok=True)
        # task_work_path should be relative to the work_base, i.e. output_dir
        task_work_path_list.append(single_calc_dir.name)

        output_file = single_calc_dir/"input.inp"
        write_cp2k_input(input_dict, output_file)
        print(
            f"Input file for efield {intensity:7.6f} written to {output_file}")
        copy_file_list(extra_forward_files, single_calc_dir)

    return task_work_path_list


def gen_task_list(command: str,
                  task_work_path_list: List[str],
                  extra_forward_files: List[str],
                  backward_files: List[str]):
    # generate task list
    task_list = []

    outlog = CP2K_LOG_FILE
    forward_files = extra_forward_files + ["input.inp"]

    for task_work_path in task_work_path_list:
        task = Task(command=command,
                    task_work_path=task_work_path,
                    forward_files=forward_files,
                    backward_files=backward_files,
                    outlog=outlog)
        task_list.append(task)
    return task_list


def calc_diel_global(input_file: str,
                     intensity_array: npt.NDArray[np.float64],
                     axis: str,
                     eps_type: str,
                     output_dir: str,
                     machine_dict: Dict,
                     resources_dict: Dict,
                     command: str,
                     extra_forward_files: List[str] = [],
                     extra_forward_common_files: List[str] = [],
                     restart_wfn: str = None,
                     dry_run: bool = False,
                     ):
    # gen input dict
    template_input_dict = gen_cp2k_input_dict(input_file, canonical=True)
    # gen task work path list

    task_work_path_list = \
        gen_series_calc_efield(input_dict=template_input_dict,
                               intensity_array=intensity_array,
                               displacement_field=False,
                               axis=axis,
                               periodic=True,
                               eps_type=eps_type,
                               filename="="+DIPOLE_MOMENT_FILE,
                               output_dir=output_dir,
                               extra_forward_files=extra_forward_files,
                               restart_wfn=restart_wfn
                               )
    # gen task
    task_list = gen_task_list(command=command,
                              task_work_path_list=task_work_path_list,
                              extra_forward_files=extra_forward_files,
                              backward_files=[
                                  DIPOLE_MOMENT_FILE, CP2K_LOG_FILE]
                              )
    # submission
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    # to absolute path
    # TODO: bug here the common files cannot be uploaded using LazyLocalContext.
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
    if dry_run:
        # dry_run has been already true
        exit_on_submit = True
    else:
        exit_on_submit = False
    submission.run_submission(dry_run=dry_run, exit_on_submit=exit_on_submit)

    dipole_moment_array = get_dipole_moment_array(
        task_work_path_list, output_dir, axis)
    volume_array = get_volume_array(task_work_path_list, output_dir)
    # use one volume for all calculation
    dielectric_constant = get_dielectric_constant(dipole_moment_array,
                                                  intensity_array,
                                                  volume_array)
    plot_dielectric_fitting(intensity_array=intensity_array,
                            dipole_moment_array=dipole_moment_array,
                            dielectric_constant=dielectric_constant,
                            output_dir=output_dir,
                            axis=axis,)
    #
    output_dir = Path(output_dir)
    np.savetxt(output_dir/"dipole_moment_array.dat", dipole_moment_array)
    np.savetxt(output_dir/"intensity_array.dat", intensity_array)
    np.savetxt(output_dir/"volume_array.dat", volume_array)

    print(f"The Dielectric Constant is {dielectric_constant:10.6f}")
    print("Workflow for Calculation of Dielectric Constant Complete!")


def calc_diel_atomic(input_file: str,
                     intensity: npt.NDArray[np.float64],
                     axis: str,
                     eps_type: str,
                     output_dir: str,
                     machine_dict: Dict,
                     resources_dict: Dict,
                     command: str,
                     extra_forward_files: List[str] = [],
                     extra_forward_common_files: List[str] = [],
                     restart_wfn: str = None,
                     dry_run: bool = False,
                     ):
    # gen input dict
    template_input_dict = gen_cp2k_input_dict(input_file, canonical=True)
    # gen task work path list
    task_work_path_list = \
        gen_calc_opposite_efield(input_dict=template_input_dict,
                                 intensity=intensity,
                                 displacement_field=False,
                                 axis=axis,
                                 periodic=True,
                                 eps_type=eps_type,
                                 filename="="+DIPOLE_MOMENT_FILE,
                                 output_dir=output_dir,
                                 extra_forward_files=extra_forward_files,
                                 restart_wfn=restart_wfn
                                 )

    # gen task
    task_list = gen_task_list(command=command,
                              task_work_path_list=task_work_path_list,
                              extra_forward_files=extra_forward_files,
                              backward_files=[DIPOLE_MOMENT_FILE,
                                              CP2K_LOG_FILE, DENSITY_FILE]
                              )
    # submission
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    # to absolute path
    # TODO: bug here the common files cannot be uploaded using LazyLocalContext.
    forward_common_files = extra_forward_common_files
    if restart_wfn:
        forward_common_files.append(restart_wfn)
    # copy to the work base directory so that it can be uploaded
    copy_file_list(forward_common_files, output_dir)
    # workbase will be coverted to absolute path
    # local_root/taskpath is the full path for upload files
    submission = Submission(work_base=output_dir,
                            machine=machine,
                            resources=resources,
                            task_list=task_list,
                            forward_common_files=forward_common_files,
                            backward_common_files=[],
                            )
    if dry_run:
        # dry_run has been already true
        exit_on_submit = True
    else:
        exit_on_submit = False
    submission.run_submission(dry_run=dry_run, exit_on_submit=exit_on_submit)

    z, dielectric_constant = get_dielectic_constant_atomic(task_work_path_list=task_work_path_list,
                                                           output_dir=output_dir,
                                                           intensity=intensity,
                                                           axis=axis
                                                           )

    output_dir = Path(output_dir)
    np.savetxt(output_dir/"diel_profile.dat",
               np.array([z, dielectric_constant]).T)
    plot_dielectric_profile(z, dielectric_constant, output_dir, axis)
    print("Workflow for Calculation of Dielectric Constant Complete!")


def calc_diel(input_dict: Dict,
              machine_dict: Dict,
              resources_dict: Dict,
              dry_run: bool = False,
              ):
    # the wrap function for different types of dielectric constant calculation

    scale = input_dict.pop('scale', 'global')
    if scale == 'global':
        calc_diel_global(**input_dict,
                         machine_dict=machine_dict,
                         resources_dict=resources_dict,
                         dry_run=dry_run)
    elif scale == 'atomic':
        calc_diel_atomic(**input_dict,
                         machine_dict=machine_dict,
                         resources_dict=resources_dict,
                         dry_run=dry_run)
    else:
        print(f"The scale {scale} is not supported!")
