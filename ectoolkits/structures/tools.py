from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdMolDescriptors

def smiles2xyz(smiles: str, filename: str):
    """
    Convert a SMILES string to a 3D XYZ structure using RDKit and ASE.

    Args:
        smiles (str): SMILES representation of the molecule.
        filename (str): Output filename in XYZ format.

    Returns:
        None

    Example:
        smiles2xyz("CCO", "ethanol.xyz")
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    atoms = []
    positions = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        atoms.append(atom.GetSymbol())
    molecule = Atoms(symbols=atoms, positions=positions)
    write(filename, molecule)
    
def get_packmol_inp(filename: str, count: int, n_vec_a: list, n_vec_b: list, 
                    d1_a: float, d2_a: float, d1_b: float, d2_b: float,
                    low_z: float, up_z: float, radius: float) -> str:
    """
    Generate PACKMOL input structure block for a molecule.

    Args:
        filename (str): Path to the XYZ file.
        count (int): Number of molecules to place.
        n_vec_a (list): Normal vector A [x, y, z].
        n_vec_b (list): Normal vector B [x, y, z].
        d1_a, d2_a (float): Distance range for vector A.
        d1_b, d2_b (float): Distance range for vector B.
        low_z, up_z (float): Z direction limits.
        radius (float): Minimum distance between molecules.

    Returns:
        str: A string block to be written in a PACKMOL input file.
    """
    return f"""
structure {filename}
  number {count}
  above plane {n_vec_a[0]:6.4f} {n_vec_a[1]:6.4f} {n_vec_a[2]:6.4f} {d1_a + 0.5:6.4f}
  below plane {n_vec_a[0]:6.4f} {n_vec_a[1]:6.4f} {n_vec_a[2]:6.4f} {d2_a - 0.5:6.4f}
  above plane {n_vec_b[0]:6.4f} {n_vec_b[1]:6.4f} {n_vec_b[2]:6.4f} {d1_b + 0.5:6.4f}
  below plane {n_vec_b[0]:6.4f} {n_vec_b[1]:6.4f} {n_vec_b[2]:6.4f} {d2_b - 0.5:6.4f}
  above plane 0.0000 0.0000 1.0000 {low_z:6.4f}
  below plane 0.0000 0.0000 1.0000 {up_z:6.4f}
  radius {radius:6.4f}
end structure
"""

def count_heavy_atoms(smiles: str) -> int:
    """
    Count the number of heavy atoms (non-hydrogen) in a SMILES string.

    Args:
        smiles (str): SMILES string.

    Returns:
        int: Number of heavy atoms.

    Raises:
        ValueError: If the SMILES string is invalid.

    Example:
        count_heavy_atoms("CCO")  # returns 3
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

def get_exclude_water_count(
        ads_smiles: list, ads_counts: list,
        bulk_smiles: list, bulk_counts: list,
        water_heavy_atom_equiv: int = 1
    ) -> int:
    """
    Estimate the number of equivalent water molecules to be excluded based on heavy atom count.

    Args:
        ads_smiles (list): List of adsorbate SMILES strings.
        ads_counts (list): List of adsorbate molecule counts.
        bulk_smiles (list): List of bulk molecule SMILES strings.
        bulk_counts (list): List of bulk molecule counts.
        water_heavy_atom_equiv (int): Number of heavy atoms equivalent to one water molecule.

    Returns:
        int: Estimated number of water molecules to exclude.

    Example:
        get_exclude_water_count(["CCO"], [2], ["O=C=O"], [1], 3)
    """
    total_heavy_atoms = 0
    for smiles, count in zip(ads_smiles, ads_counts):
        total_heavy_atoms += count * count_heavy_atoms(smiles)
    for smiles, count in zip(bulk_smiles, bulk_counts):
        total_heavy_atoms += count * count_heavy_atoms(smiles)
    return total_heavy_atoms // water_heavy_atom_equiv      
