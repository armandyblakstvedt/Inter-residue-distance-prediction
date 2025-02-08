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
    coords = np.array(coords)

    # Compute the full distance map if at least one coordinate is available
    if len(coords) >= 1:
        if len(coords) > 1:
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            full_distance_map = np.sqrt(np.sum(diff**2, axis=-1))
            n_atoms = full_distance_map.shape[0]
            # Extract upper triangle (including the diagonal) as a flat array
            upper_triangle = full_distance_map[np.triu_indices(n_atoms)]
        else:
            upper_triangle = np.array([0.0])
    else:
        upper_triangle = np.empty((0,))

    # Store the actual number of residues with C-alpha atoms before padding.
    n_actual = len(sequence)
    # Standardize sequence and pad the upper-triangle to fixed length 80200
    if n_actual > 400 or n_actual < 300:
        return None
    sequence = sequence + ['Ø'] * (400 - n_actual)
    fixed_len = 80200
    padded_dm = np.pad(upper_triangle, (0, fixed_len - upper_triangle.size), constant_values=np.nan)

    # Return n_actual so we know which part of the matrix has valid data.
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


def reconstruct_symmetric_matrix(upper_vector, n, size=400):
    """Reconstruct a symmetric matrix from the flattened upper-triangle, filling the remaining cells with NaN."""
    matrix = np.full((size, size), np.nan)  # fill all with NaN
    idx = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = upper_vector[idx]
            matrix[j, i] = upper_vector[idx]
            idx += 1
    return matrix


def main():
    data = load_data()  # Each element: [sequence, padded_dm, n_actual]
    if len(data) == 0:
        print("No data to display.")
        exit(1)
    print(data)
    current_idx = [0]
    fig, ax = plt.subplots()

    # Reconstruct the full symmetric distance matrix for visualization.
    n_actual = data[current_idx[0]][2]
    matrix = reconstruct_symmetric_matrix(data[current_idx[0]][1], n_actual)
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
        n_actual = data[current_idx[0]][2]
        matrix = reconstruct_symmetric_matrix(data[current_idx[0]][1], n_actual)
        img.set_data(matrix)
        ax.set_title(f"Map {current_idx[0]+1}/{len(data)}")
        fig.canvas.draw_idle()

    ax.set_title(f"Map {current_idx[0]+1}/{len(data)}")
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == '__main__':
    main()
