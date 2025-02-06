import glob
from Bio.PDB import PDBParser
import numpy as np

# Use glob to get all .pdb files in pdb_files/ directory
pdb_files = glob.glob("data/*.pdb")
parser = PDBParser()

sequences = []
# Iterate over each PDB file
for pdb_file in pdb_files:
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

    # Append the sequence and distance map to the sequences list
    sequences.append(["".join(sequence), distance_map])

# Print the average length of the sequences
avg_length = np.mean([len(seq[0]) for seq in sequences])
print(f"Average sequence length: {avg_length:.2f}")

# Print the max length of the sequences and the min length
max_length = max([len(seq[0]) for seq in sequences])
print(f"Max sequence length: {max_length}")
min_length = min([len(seq[0]) for seq in sequences])
print(f"Min sequence length: {min_length}")

# Print how many fall outside the sequence length of 400 to 500
outside_range = sum([len(seq[0]) < 400 or len(seq[0]) > 500 for seq in sequences])
print(f"Sequences outside the range of 400 to 500: {outside_range}")

# Print how many fall between the sequence length of 400 to 500
inside_range = sum([len(seq[0]) >= 400 and len(seq[0]) <= 500 for seq in sequences])
print(f"Sequences inside the range of 400 to 500: {inside_range}")

# Print all the different amino acids found in the sequences
all_amino_acids = set()
for seq in sequences:
    all_amino_acids.update(seq[0])
print(f"All amino acids found: {all_amino_acids}")
