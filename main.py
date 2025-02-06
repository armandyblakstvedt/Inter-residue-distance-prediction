import torch
import tqdm
from torch.amp import autocast
from utils.load_data import load_data
from utils.one_hot_encode import one_hot_encode
from sklearn.preprocessing import MinMaxScaler
from ProteinDataset import ProteinDataset
from torch.utils.data import DataLoader
from model import Model

EPOCHS = 10

NUMBER_OF_BATCHES_PER_EPOCH = 100


def train(model, dataloader, criterion, optimizer, scaler_grad):
    model.train()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(dataloader)):
            if NUMBER_OF_BATCHES_PER_EPOCH is not None and batch_idx >= NUMBER_OF_BATCHES_PER_EPOCH:
                break

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                output = model(data)
                loss = criterion(output, target)

            scaler_grad.scale(loss).backward()

            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()

            running_loss += loss.item()

        epoch_loss = running_loss / batch_idx

        scheduler.step(epoch_loss)


def predict(model, dataloader):
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm.tqdm(dataloader)):
            if NUMBER_OF_BATCHES_PER_EPOCH is not None and batch_idx >= NUMBER_OF_BATCHES_PER_EPOCH:
                break

            output = model(data)
            predictions.append(output)

    return predictions


if __name__ == '__main__':
    data = load_data()

    # Extract all the unique amino acids in the dataset
    all_amino_acids = set()
    for seq, _ in data:
        all_amino_acids.update(seq)

    # One hot encode the protein sequence
    # the data is a tuple of (protein_sequence, target)
    # the protein_sequence is a string of amino acids
    # the target is a inter-residue distance matrix
    # the target is a 2D numpy array
    data = [(one_hot_encode(seq, all_amino_acids), target) for seq, target in data]

    dataset = ProteinDataset(data)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model()

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler_grad = torch.amp.grad_scaler.GradScaler()

    train(model, dataloader, criterion, optimizer, scaler_grad)
