import pytest 
import toolkit
from ase.io import read

from toolkit.structures.rutile110 import (Rutile110, 
                                          Rutile1p11Edge)


#* Test flat structures

flat_file_list = [
    "../_structures/4x2-flat-alongy.cif",
    "../_structures/8x4-flat-alongy.cif"
]

arg_list       = [
    [2, 'y'],
    [4, 'y'],
]

flat_result_list = 