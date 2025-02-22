from ase.build import sort
from ase.io import read,vasp
import os

# Displacement from equilibrium
displacement=0.1
cart=['x','y','z']

# Loop over Cartesian directions, positive displacement
for i in range(3):
    # Loop over directions
    for direction in ["plus","minus"]:
        # Read file
        at0 = sort(read('CONTCAR'))
        # Copy positions
        positions=at0.get_positions()
        # Displace H atom (the last one) from equilibrium
        if direction=="plus":
            positions[-1][i]+=displacement
        else:
            positions[-1][i]-=displacement
        # Update positions
        at0.set_positions(positions)
        # Make directory
        dirname="displ_"+cart[i]+"_"+direction
        os.makedirs(dirname,exist_ok=True)
        # Write the structure to a VASP POSCAR file
        vasp.write_vasp(dirname+'/POSCAR', at0, vasp5=True, direct=True)


