from scipy.ndimage import gaussian_filter1d
from utils import au2eV
import numpy as np

def read_dos_element(file):
    with open(file) as f:
        first_line = f.readline()
        element = first_line.split()[6]
    return element

def read_dos_fermi(file):
    with open(file) as f:
        first_line = f.readline()
        fermi = first_line.split()[15]
        fermi = float(fermi)*au2eV
    return fermi

def read_dos_energies(file):
    energies = np.loadtxt(file, usecols=1)
    energies = energies * au2eV
    return energies

def get_raw_dos(energies, fermi, weights, step=0.1):
    bins = int((energies[-1]-energies[0])/step)
    dos, ener = np.histogram(energies, bins, weights=weights)
    ener = ener[:-1] - fermi + 0.5*step
    return dos, ener

def dos_smth(dos, sigma=0.2):
    #smooth the dos data
    smth_dos = gaussian_filter1d(dos, sigma)
    return smth_dos

def get_dos(file, dos_type="total"):
    # get the dos from cp2k pdos file
    # dos_type: total, s, p, d
    # return information, dos, ener
    info = {}
    info['element'] = read_dos_element(file)
    info['fermi'] = read_dos_fermi(file)
    # read energy
    energies = read_dos_energies(file)
    # decide the weight from dos_type
    if dos_type == "total":
        tmp_len = len(np.loadtxt(file, usecols = 2))
        weights = np.ones(tmp_len)
    elif dos_type == "s":
        weights = np.loadtxt(file, usecols = 3)
    elif dos_type == "p":
        weights = np.loadtxt(file, usecols = (4,5,6)).sum(axis=1)
    elif dos_type == "d":
        weights = np.loadtxt(file, usecols = (7,8,9,10,11)).sum(axis=1)
    elif dos_type == "f":
        weights = np.loadtxt(file, usecols = (12,13,14,15,16,17,18)).sum(axis=1)
    else:
        raise NameError("dos type does not exist!")
    # make dos by histogram
    dos, ener = get_raw_dos(energies, info['fermi'], weights)
    # smooth
    smth_dos = dos_smth(dos)
    return smth_dos, ener, info
