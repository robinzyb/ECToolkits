from ase import Atoms
from ectoolkits.structures.slab import Slab
import numpy as np


class Interface(Slab):

    def __init__(self, atoms, *args, **kwargs):
        super().__init__(atoms)
