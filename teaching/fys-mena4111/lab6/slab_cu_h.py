from ase.build import fcc111, add_adsorbate
from ase.io import vasp
from ase import Atoms

# Define the lattice constant
a_latt = 3.64

# Define the adsorbate atom and height above the surface layer
adsorbate = Atoms('H')
height = 1.0
# Define the adsorption site. Supported special adsorption sites for fcc111: 'ontop', 'bridge', 'fcc', 'hcp'.
site = 'fcc'

# Build a 111 slab with 4 layers and a vacuum layer of 12 AA
slab = fcc111('Cu', (1, 1, 4), a=a_latt, vacuum=6.0)

# Add the adsorbate atom to the slab
add_adsorbate(slab, adsorbate, height, site)

# Write the structure to a VASP POSCAR file
vasp.write_vasp('POSCAR', slab, vasp5=True, direct=True)

