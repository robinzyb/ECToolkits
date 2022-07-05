from ase import Atoms
from slab import Slab
import numpy as np

class Interface(Slab):

    def __init__(self, atoms, *args, **kwargs):
        super().__init__(atoms)

    def placeholder(self):
        print("Child class of ase.Atoms")