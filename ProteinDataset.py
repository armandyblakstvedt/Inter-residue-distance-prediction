from torch.utils.data import Dataset
from Bio.PDB import PDBParser
import numpy as np
import torch


class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        # Ensure correct shape: [1, 10800]
        # If x does not have 10800 elements, you'll need to pad or chop it.
        x = x.view(1, -1)
        # Make Y a torch tensor
        y = torch.tensor(y, dtype=torch.float32)
        # Flatten Y
        y = y.view(-1)
        return x, y
