import glob
from Bio.PDB import PDBParser
import numpy as np


# Use glob to get all .pdb files in pdb_files/ directory
pdb_files = glob.glob("data/*.pdb")
parser = PDBParser()

# Iterate over each PDB file
for pdb_file in pdb_files:
    print(f"Processing {pdb_file}")
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Use the first model

    # Extract C-alpha atoms coordinates and sequence
    coords = []
    sequence = []
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())
                try:
                    sequence.append(residue.get_resname())
                except KeyError:
                    sequence.append("X")
    coords = np.array(coords)

    # Compute pairwise distance map if there are at least 2 coordinates
    if len(coords) > 1:
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_map = np.sqrt(np.sum(diff**2, axis=-1))
        print("Distance map:")
        print(distance_map)
    else:
        print("Not enough C-alpha coordinates to compute a distance map.")

    # Print the amino acid sequence
    print("Sequence:", "".join(sequence))
    print("="*50)
