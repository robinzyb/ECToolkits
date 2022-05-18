from ase import Atoms
import numpy as np

class Interface(Atoms):

    def __init__(self, atoms, *args, **kwargs):
        super().__init__(atoms)

    def placeholder(self):
        print("Child class of ase.Atoms")