from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from utils.feature_matrix import get_feature_matrix
import numpy as np


class ProteinDataset(Dataset):
    def __init__(self, data, dimension):
        self.data = data
        self.dimension = dimension

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        valid_entries = self.data[idx][2]
        dimension = self.dimension

        sequence, target = get_feature_matrix(x, y, dimension)

        return sequence, target, valid_entries
