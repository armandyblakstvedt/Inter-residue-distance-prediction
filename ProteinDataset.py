from torch.utils.data import Dataset
from Bio.PDB import PDBParser
import numpy as np


class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]

        return x, y
