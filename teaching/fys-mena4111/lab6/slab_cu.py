from ase.build import fcc111
from ase.io import vasp

# Define the lattice constant
a_latt = 3.64

# Build a 111 slab with 4 layers and a vacuum layer of 12 AA
slab = fcc111('Cu', (1, 1, 4), a=a_latt, vacuum=6.0)

# Write the structure to a VASP POSCAR file
vasp.write_vasp('POSCAR', slab, vasp5=True, direct=True)
