import os
import json
import glob
import numpy as np

def get_iter_dir(dpgen_dir, num):
    """get iteration dir"""
    iter_dir = os.path.join(dpgen_dir, "iter.{0:06d}".format(num))
    return iter_dir

def get_devi_dir(iter_dir):
    """get model deviation dir"""
    devi_dir = os.path.join(iter_dir, "01.model_devi")
    return devi_dir

def get_task_dir(devi_dir, sys_idx, task_idx):
    """get md task dir"""
    task_dir = os.path.join(devi_dir, "task.{0:03d}.{1:06d}".format(sys_idx, task_idx))
    return task_dir

def get_md_init_num(sys_config, devi_setup):
    """get md initial structure number"""
    md_num = 0
    for sys_idx in devi_setup["sys_idx"]:
        for structs in sys_config[sys_idx]:
            md_num += len(glob.glob(structs))
    if md_num == 0:
        raise ValueError("Can't find any structs")
    else:
        return md_num

def get_param_path(dpgen_dir, param):
    """get the parameter json file path"""
    return os.path.join(dpgen_dir, param)

def get_model_devi_path(task_dir):
    """get the model deviation file path"""
    model_devi_path = os.path.join(task_dir, "model_devi.out")
    return model_devi_path

def get_max_force_devi(model_devi_path):
    """
    get the max force deviation from out file
    """
    force_devi = np.loadtxt(model_devi_path, usecols=4)
    return force_devi

def store_devi_data(model_devi_data, force_devi, temp, i):
    """
    store the deviation force into a dictionary
    """
    if temp in model_devi_data:
        if "{0:06d}".format(i) in model_devi_data[temp]:
            model_devi_data[temp]["{0:06d}".format(i)].append(force_devi)
        else:
            model_devi_data[temp]["{0:06d}".format(i)] = []
            model_devi_data[temp]["{0:06d}".format(i)].append(force_devi)
    else:
        model_devi_data[temp] = {}
        model_devi_data[temp]["{0:06d}".format(i)] = []
        model_devi_data[temp]["{0:06d}".format(i)].append(force_devi)

def collect_model_devi(dpgen_dir, param_file, iteration):
    """
    main function to collect model deviation result
    dpgen_dir: the dir containning iter...
    param_file: the name of param.json
    iteration: collect model deviation data before this iteration
    """
    # open the param json file
    param_file = get_param_path(dpgen_dir, param_file)
    with open(param_file) as f:
        jdata = json.load(f)
    # all data store in this dictionary
    model_devi_data = {}
    for i in range(iteration):
        # get the dir now, i is iteration number
        iter_dir = get_iter_dir(dpgen_dir, i)
        devi_dir = get_devi_dir(iter_dir)
        # get the current deviation setup
        devi_setup = jdata["model_devi_jobs"][i]
        # get the md initial structure number
        sys_config = jdata["sys_configs"]
        md_num = get_md_init_num(sys_config, devi_setup)
        #print(md_num)
        # get the task dir
        # count is current task number
        count = 0
        for j in range(md_num):
            for temp in devi_setup["temps"]:
                for sys_idx in devi_setup["sys_idx"]:
                    task_dir = get_task_dir(devi_dir, sys_idx, count)
                    #get the model deviation .out path
                    model_devi_file = get_model_devi_path(task_dir)
                    max_force_devi = get_max_force_devi(model_devi_file)
                    #add to the dictionary
                    store_devi_data(model_devi_data, max_force_devi, temp, i)
                    print("temperature: {0}K \n trajectory directory: {1}".format(temp, task_dir))
                    count += 1
    return model_devi_data
