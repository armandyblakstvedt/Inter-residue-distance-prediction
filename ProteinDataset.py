from torch.utils.data import Dataset
from Bio.PDB import PDBParser
import numpy as np


class ProteinDataset(Dataset):
    def __init__(self, pdb_files):
        self.pdb_files = pdb_files
        self.parser = PDBParser()
        self.sequences = self._get_sequences()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def _get_sequences(self):
        sequences = []
        for pdb_file in self.pdb_files:
            structure = self.parser.get_structure("protein", pdb_file)
            model = structure[0]

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

            if len(coords) > 1:
                diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
                distance_map = np.sqrt(np.sum(diff**2, axis=-1))

            sequences.append(["".join(sequence), distance_map])

        return sequences
