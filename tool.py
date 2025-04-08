from Bio import SeqIO
import torch
import matplotlib.pyplot as plt
from ProteinDataset import ProteinDataset
from model import Model
from utils.load_data import one_hot_encode
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionary to map single-letter amino acid codes to three-letter codes
d = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS',
     'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN',
     'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP',
     'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET'}

def predict(record, model):
    # Create a ProteinDataset instance
    sequence = record.seq

    # Throw error if the sequence is over 400 characters
    if len(sequence) > 400:
        raise ValueError("Sequence length exceeds 400 characters.")

    # Convert the sequence of single-letter codes to a sequence of three-letter codes
    sequence = [str(aa) for aa in sequence]

    # Convert the sequence to a list of three-letter codes
    sequence = [d[aa] for aa in sequence if aa in d]

    # Add 'Ø' to the sequence until it reaches 400 characters
    sequence += ['Ø'] * (400 - len(sequence))

    data = [(sequence, torch.zeros(400, 400), 400)]
    one_hot_encoded_data, num_unique_amino_acids = one_hot_encode(data, DEVICE)

    protein_dataset = ProteinDataset(one_hot_encoded_data, num_unique_amino_acids)
    dataloader = torch.utils.data.DataLoader(protein_dataset, batch_size=1, shuffle=False)

    sample_data, _, _ = next(iter(dataloader))

    with torch.no_grad():
        prediction = model(sample_data)

    return prediction[0, 0].cpu().numpy()


def read_fasta(filepath: str):
    return SeqIO.parse(filepath, "fasta")

def load_model(model_path: str):
    model = Model(
        input_channels=2 * 86,
    )

    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model.to(DEVICE)

def main():
    # Load the model
    model = load_model("model.pth")

    # Input with autocomplete for file path
    fasta_path = input("Enter the path to the FASTA file: ")
    fasta_records = read_fasta(fasta_path)
    for record in fasta_records:
        width = len(record.seq)
        height = len(record.seq)

        prediction_matrix = predict(record, model)

        # Cut off the prediction matrix to the size of the sequence
        prediction_matrix = prediction_matrix[:width, :height]

        # Plot the prediction
        plt.imshow(prediction_matrix, cmap='viridis', interpolation='nearest')

        # Color represents the predicted distance in Angstroms (Å)
        plt.colorbar(label='Predicted Distance (Å)')
        plt.title(f"Prediction for {record.id}")

        # Save the plot
        os.makedirs("plots", exist_ok=True)
        filename_prefix = record.id.replace('|', '-').replace(' ', '-') # Replace slashes and spaces in the filename
        plt.savefig(f"plots/{filename_prefix}_prediction.png")
        print(f"Plot saved for {record.id} at path 'plots/{filename_prefix}_prediction.png'")

        plt.show()
        plt.close()

        # Save the prediction matrix to a pickle file
        os.makedirs("predictions", exist_ok=True)
        with open(f"predictions/{filename_prefix}_prediction.pkl", "wb") as f:
            torch.save(prediction_matrix, f)
        print(f"Prediction saved for {record.id} at path 'predictions/{filename_prefix}_prediction.pkl'")

if __name__ == "__main__":
    main()