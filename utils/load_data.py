import glob
import os
import pickle
from Bio.PDB import PDBParser
import numpy as np
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch

warnings.simplefilter('ignore', PDBConstructionWarning)
parser = PDBParser()

CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "sequences.pkl")
CACHED_ONE_HOT = os.path.join(CACHE_DIR, "one_hot_encoded_data.pt")
CACHED_UNIQUE_AMINO_ACIDS = os.path.join(CACHE_DIR, "unique_amino_acids.pkl")

def process_file(pdb_file):
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

    # Compute full distance map from coordinates
    if len(coords) >= 1:
        if len(coords) > 1:
            coords = np.array(coords)
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            full_distance_map = np.sqrt(np.sum(diff**2, axis=-1))
        else:
            full_distance_map = np.array([[0.0]])
    else:
        full_distance_map = None

    n_actual = len(sequence)
    # Only process proteins with 300 to 400 C-alpha residues.
    if n_actual > 400 or n_actual < 300:
        return None
    # Pad the sequence to length 400 using a filler character
    sequence = sequence + ['Ø'] * (400 - n_actual)

    # Pad the computed distance matrix into a fixed-size 400x400 matrix with NaN.
    fixed_size = 400
    padded_dm = np.full((fixed_size, fixed_size), np.nan)
    if full_distance_map is not None:
        padded_dm[:n_actual, :n_actual] = full_distance_map

    return [sequence, padded_dm, n_actual]


def load_data():
    pdb_files = glob.glob("data/*.pdb")
    sequences = []
    with ProcessPoolExecutor() as executor:
        results = tqdm(executor.map(process_file, pdb_files), total=len(pdb_files), desc="Processing files")
        for result in results:
            if result is not None:
                sequences.append(result)
    return sequences

def get_unique_amino_acids(DEVICE="cpu"):
    if os.path.exists(CACHED_UNIQUE_AMINO_ACIDS):
        print("Loading cached amino acids...")
        amino_acid_data = torch.load(CACHED_UNIQUE_AMINO_ACIDS, map_location=DEVICE)
        return amino_acid_data.get("aa_to_index"), amino_acid_data.get("num_unique_amino_acids")

    # Get data based on sequences in cache
    if os.path.exists(CACHE_FILE):
        print("Loading cached data...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Cache not found. Processing data...")
        data = load_data()
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)

    # Extract all the unique amino acids in the dataset
    print("Extracting unique amino acids...")
    all_amino_acids = set()
    for seq, _, _ in tqdm(data, desc="Extracting unique amino acids", unit="protein"):
        for aa in seq:
            all_amino_acids.add(aa)
    all_amino_acids = sorted(list(all_amino_acids))
    aa_to_index = {aa: idx for idx, aa in enumerate(all_amino_acids)}
    num_unique_amino_acids = len(all_amino_acids)

    # Save to cache
    amino_acid_data = {
        "aa_to_index": aa_to_index,
        "num_unique_amino_acids": num_unique_amino_acids
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(amino_acid_data, CACHED_UNIQUE_AMINO_ACIDS)
    print("Unique amino acids cached.")

    return aa_to_index, num_unique_amino_acids

def one_hot_encode(data, DEVICE="cpu"):
    """
    One-hot encode the protein sequences and convert distance matrices to tensors.
    Args:
        data: List of tuples, where each tuple contains:
            - sequence: List of amino acids (strings).
            - padded_dm (Optional, required for training): Numpy array of the distance matrix.
            - n_actual (Optional, required for training): Number of actual residues in the sequence.
        DEVICE: The device to which the tensors will be moved (CPU or GPU).
    Returns:
        one_hot_encoded_data: List of tuples, where each tuple contains:
            - one_hot: Tensor of shape (400 * num_unique_amino_acids).
            - padded_dm: Tensor of shape (400, 400).
            - n_actual: Tensor of shape (1,).
        num_unique_amino_acids: The number of unique amino acids in the dataset.
    """
    aa_to_index, num_unique_amino_acids = get_unique_amino_acids(DEVICE)

    one_hot_encoded_data = []
    for protein_sequence, distance_matrix, valid_entries in tqdm(
            data, desc='One-hot encoding', unit='protein'):
        one_hot = torch.zeros(len(protein_sequence), num_unique_amino_acids, dtype=torch.float32, device=DEVICE)
        for i, aa in enumerate(protein_sequence):
            one_hot[i, aa_to_index[aa]] = 1
        one_hot = one_hot.view(-1)
        one_hot_encoded_data.append((
            one_hot,
            torch.tensor(distance_matrix, dtype=torch.float32, device=DEVICE),
            torch.tensor(valid_entries, dtype=torch.float32, device=DEVICE)
        ))

    return one_hot_encoded_data, num_unique_amino_acids

def load_cached_data(DEVICE):
    if os.path.exists(CACHE_FILE):
        print("Loading cached data...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Cache not found. Processing data...")
        data = load_data()
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)

    # Check for cached one-hot encoded data
    if os.path.exists(CACHED_ONE_HOT):
        print("Loading cached one-hot encoded data...")
        one_hot_encoded_data = torch.load(CACHED_ONE_HOT, map_location=DEVICE)
        num_unique_amino_acids = one_hot_encoded_data[0][0].shape[0] // torch.sum(one_hot_encoded_data[0][0] == 1).item()
    else:
        one_hot_encoded_data, num_unique_amino_acids = one_hot_encode(data, DEVICE)
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(one_hot_encoded_data, CACHED_ONE_HOT)
        print("One-hot encoded data cached.")

    return one_hot_encoded_data, num_unique_amino_acids


def main():
    data = load_data()  # Each element: [sequence, padded_dm, n_actual]
    if len(data) == 0:
        print("No data to display.")
        exit(1)
    print(data)
    current_idx = [0]
    fig, ax = plt.subplots()

    # Use the full distance matrix directly for visualization.
    matrix = data[current_idx[0]][1]
    img = ax.imshow(matrix, aspect='auto')
    plt.colorbar(img)

    def on_key(event):
        nonlocal matrix
        if event.key == 'right':
            current_idx[0] = (current_idx[0] + 1) % len(data)
        elif event.key == 'left':
            current_idx[0] = (current_idx[0] - 1) % len(data)
        else:
            return
        matrix = data[current_idx[0]][1]
        img.set_data(matrix)
        ax.set_title(f"Map {current_idx[0]+1}/{len(data)}")
        fig.canvas.draw_idle()

    ax.set_title(f"Map {current_idx[0]+1}/{len(data)}")
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == '__main__':
    main()
