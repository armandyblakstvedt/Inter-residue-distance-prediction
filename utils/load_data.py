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

warnings.simplefilter('ignore', PDBConstructionWarning)
parser = PDBParser()

CACHE_FILE = 'cache/sequences.pkl'


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


def load_cached_data():
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
    return data


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
