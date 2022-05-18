from ase import Atoms
import numpy as np

class Interface(Atoms):

    def __init__(self, atoms, *args, **kwargs):
        super().__init__(atoms)

    def placeholder(self):
        print("Child class of ase.Atoms")


dm = np.array([[ 0.,  1.,  1., 12.,  7.,  6.,  6.,  5.],
               [ 1.,  0.,  0., 13.,  8.,  7.,  7.,  6.],
               [ 1.,  0.,  0., 13.,  8.,  7.,  7.,  6.],
               [12., 13., 13.,  0.,  5.,  6.,  6.,  7.],
               [ 7.,  8.,  8.,  5.,  0.,  1.,  1.,  2.],
               [ 6.,  7.,  7.,  6.,  1.,  0.,  0.,  1.],
               [ 6.,  7.,  7.,  6.,  1.,  0.,  0.,  1.],
               [ 5.,  6.,  6.,  7.,  2.,  1.,  1.,  0.]])