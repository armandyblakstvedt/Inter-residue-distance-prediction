import torch
import tqdm
from torch.amp import autocast
from utils.load_data import load_data
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

    # Convert the set to a list and sort for consistent order
    all_amino_acids = sorted(list(all_amino_acids))

    # Create a mapping from amino acid to index
    aa_to_index = {aa: idx for idx, aa in enumerate(all_amino_acids)}
    num_amino_acids = len(all_amino_acids)

    # One hot encode the data
    one_hot_encoded_data = []
    for seq, target in data:
        # If target is longer than 400, truncate it
        if len(seq) > 400:
            print(len(seq))
            exit(0)
        # Create a tensor filled with zeros of shape (400, num_amino_acids)
        one_hot = torch.zeros(len(seq), num_amino_acids)
        for i, aa in enumerate(seq):
            one_hot[i, aa_to_index[aa]] = 1
        # Flatten the one hot tensor to get a vector of length num_amino_acids * 400
        one_hot = one_hot.view(-1)
        one_hot_encoded_data.append((one_hot, target))

    print(num_amino_acids)

    print(one_hot_encoded_data[0][0].shape)

    exit(0)

    data = one_hot_encoded_data

    dataset = ProteinDataset(data)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model(400 * 400)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler_grad = torch.amp.grad_scaler.GradScaler()

    train(model, dataloader, criterion, optimizer, scaler_grad)
