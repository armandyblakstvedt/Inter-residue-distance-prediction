import torch
import tqdm
from torch.amp import autocast
from utils.load_data import load_data
from ProteinDataset import ProteinDataset
from torch.utils.data import DataLoader
from model import Model
from utils.losses import MaskedMSELoss

EPOCHS = 10

NUMBER_OF_BATCHES_PER_EPOCH = None


def train(model, dataloader, criterion, optimizer, scaler_grad):
    model.train()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Another scheduler that can be used is ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in tqdm.tqdm(range(EPOCHS), desc='Epochs', unit='epoch'):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            if NUMBER_OF_BATCHES_PER_EPOCH is not None and batch_idx >= NUMBER_OF_BATCHES_PER_EPOCH:
                break

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            scaler_grad.scale(loss).backward()

            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()

            running_loss += loss.item()

        epoch_loss = running_loss / batch_idx

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

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
        # print(seq)
        for aa in seq:
            all_amino_acids.add(aa)

    # Convert the set to a list and sort for consistent order
    all_amino_acids = sorted(list(all_amino_acids))

    # Create a mapping from amino acid to index
    aa_to_index = {aa: idx for idx, aa in enumerate(all_amino_acids)}
    num_unique_amino_acids = len(all_amino_acids)

    # One hot encode the data
    one_hot_encoded_data = []
    for protein_sequence, distance_matrix in data:
        # Create a tensor filled with zeros of shape (400, num_amino_acids)
        one_hot = torch.zeros(len(protein_sequence), num_unique_amino_acids)
        for i, aa in enumerate(protein_sequence):
            one_hot[i, aa_to_index[aa]] = 1
        # Flatten the one hot tensor to get a vector of length num_amino_acids * 400
        one_hot = one_hot.view(-1)
        one_hot_encoded_data.append((one_hot, distance_matrix))

    data = one_hot_encoded_data

    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    dataset = ProteinDataset(data)
    validation_dataset = ProteinDataset(test_data)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = Model()

    # criterion = torch.nn.MSELoss()

    criterion = MaskedMSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    scaler_grad = torch.amp.grad_scaler.GradScaler()

    train(model, dataloader, criterion, optimizer, scaler_grad)

    # Calculate the loss for the validation predictions
    validation_loss = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_dataloader):
            if NUMBER_OF_BATCHES_PER_EPOCH is not None and batch_idx >= NUMBER_OF_BATCHES_PER_EPOCH:
                break

            output = model(data)
            loss = criterion(output, target)
            validation_loss += loss.item()
            num_batches += 1

    if num_batches > 0:
        validation_loss /= num_batches
    else:
        validation_loss = float('nan')

    print(f"Validation Loss: {validation_loss:.4f}")
