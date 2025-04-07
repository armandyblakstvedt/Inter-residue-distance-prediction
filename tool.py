from Bio import SeqIO
import torch

from ProteinDataset import ProteinDataset
from model import Model
from utils.load_data import one_hot_encode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(record, model):
    # Create a ProteinDataset instance
    sequence = record.seq

    data = [sequence, None, None]
    one_hot_encoded_data, num_unique_amino_acids = one_hot_encode(data, DEVICE)

    with torch.no_grad():
        prediction = model()


def read_fasta(filepath: str):
    return SeqIO.parse(filepath, "fasta")

def load_model(model_path: str):
    model = Model(
        input_channels=2 * 86,
    )

    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

def main():
    # Load the model
    model = load_model("model.pth")

    fasta_text = input("Enter the FASTA file path: ")
    fasta_records = read_fasta(fasta_text)
    for record in fasta_records:
        prediction = predict(record, model)

if __name__ == "__main__":
    main()