import glob
from Bio.PDB import PDBParser
import numpy as np
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', PDBConstructionWarning)


def load_data():

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

        # Compute distance map if at least one coordinate is available
        if len(coords) >= 1:
            if len(coords) > 1:
                diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
                distance_map = np.sqrt(np.sum(diff**2, axis=-1))
            else:
                distance_map = np.array([[0]])
        else:
            # In case no coordinate is found, initialize an empty distance map
            distance_map = np.empty((0, 0))

        # Standardize sequence and distance matrix to length 400
        n = len(sequence)
        if n > 400:
            # Skip sequences longer than 400
            continue
        if n < 300:
            continue
        if n < 400:
            # Pad sequence with Ø
            sequence = sequence + ['Ø'] * (400 - n)
            # Create a 400x400 matrix filled with NaN and copy original distances
            padded_dm = np.full((400, 400), np.nan)
            padded_dm[:n, :n] = distance_map
            distance_map = padded_dm

        # Append the joined sequence string and its corresponding distance map
        sequences.append([sequence, distance_map])

    return sequences.copy()


if __name__ == '__main__':
    data = load_data()

    # (sequence, distances)[]
    # print(data[0][1])

    # Initialize the first image
    current_idx = [0]  # Using a list to allow modifications in the inner function

    fig, ax = plt.subplots()
    img = ax.imshow(data[current_idx[0]][1])
    plt.colorbar(img)

    def on_key(event):
        if event.key == 'right':
            current_idx[0] = (current_idx[0] + 1) % len(data)
        elif event.key == 'left':
            current_idx[0] = (current_idx[0] - 1) % len(data)
        else:
            return
        # Update image data and redraw
        img.set_data(data[current_idx[0]][1])
        ax.set_title(f"Map {current_idx[0]+1}/{len(data)}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.set_title(f"Map {current_idx[0]+1}/{len(data)}")
    plt.show()
