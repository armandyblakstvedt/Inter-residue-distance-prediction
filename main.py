import torch
from torch.amp import autocast
from utils.load_data import load_data, load_cached_data, reconstruct_symmetric_matrix
from ProteinDataset import ProteinDataset
from torch.utils.data import DataLoader
from model import Model
from utils.losses import MaskedMSELoss
import os
from utils.visualize import visualize_distances
import numpy as np
import matplotlib.pyplot as plt  # New import
from tqdm import tqdm

# Enable interactive mode for live updating plots
plt.ion()

# Torch configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Training parameters
EPOCHS = 10
NUMBER_OF_BATCHES_PER_EPOCH = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 5

# Training options
OPTIMIZER = torch.optim.Adam
SCALER_GRAD = torch.amp.grad_scaler.GradScaler()


def live_visualize(prediction, target):
    plt.figure("Prediction vs Target")
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title("Prediction")
    plt.imshow(prediction, cmap='viridis')

    plt.subplot(1, 2, 2)
    plt.title("Target")
    plt.imshow(target, cmap='viridis')

    plt.pause(0.001)


def train(model, train_dataloader, val_dataloader, criterion, optimizer, scaler_grad):
    model.train()
    # Scheduler based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(EPOCHS), desc='Epochs', unit='epoch'):
        model.train()
        running_loss = 0.0
        num_train_batches = 0

        # Training loop (scaler_grad applied on training loss)
        for batch_idx, (data, target, valid_entries) in enumerate(train_dataloader):
            if NUMBER_OF_BATCHES_PER_EPOCH is not None and batch_idx >= NUMBER_OF_BATCHES_PER_EPOCH:
                break

            optimizer.zero_grad()
            with autocast(device_type=DEVICE, dtype=torch.float32):
                output = model(data)
                loss = criterion(output, target)

            scaler_grad.scale(loss).backward()
            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()

            running_loss += loss.item()
            num_train_batches += 1

        train_loss = running_loss / num_train_batches

        # Validation loop after each epoch
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch_idx, (data, target, valid_entries) in enumerate(val_dataloader):
                if NUMBER_OF_BATCHES_PER_EPOCH is not None and batch_idx >= NUMBER_OF_BATCHES_PER_EPOCH:
                    break

                with autocast(device_type=DEVICE, dtype=torch.float32):
                    output = model(data)
                    loss = criterion(output, target)
                val_loss += loss.item()
                num_val_batches += 1

        if num_val_batches > 0:
            val_loss /= num_val_batches
        else:
            val_loss = float('nan')

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Live visualization using one sample from the validation set.
        sample_data, sample_target, sample_valid = next(iter(val_dataloader))
        # Run a forward pass on the sample
        with torch.no_grad():
            sample_pred = model(sample_data)
        # Assume valid_entries indicates the actual protein length.
        n_actual = int(sample_valid.item())
        # Crop the predicted matrix (model output has shape [batch, 1, H, W])
        pred_matrix = sample_pred[0, 0].cpu().numpy()
        pred_matrix = pred_matrix[:n_actual, :n_actual]
        # Reconstruct the target symmetric matrix using the helper.
        target_vector = sample_target[0].cpu().numpy()
        target_matrix = reconstruct_symmetric_matrix(target_vector, n_actual)
        live_visualize(pred_matrix, target_matrix)

        # Check for improvement for early stopping (using validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

        # Step scheduler based on validation loss
        scheduler.step(val_loss)


if __name__ == '__main__':
    data = load_cached_data()

    # Extract all the unique amino acids in the dataset
    all_amino_acids = set()
    for seq, _, _ in data:
        for aa in seq:
            all_amino_acids.add(aa)
    all_amino_acids = sorted(list(all_amino_acids))
    aa_to_index = {aa: idx for idx, aa in enumerate(all_amino_acids)}
    num_unique_amino_acids = len(all_amino_acids)

    one_hot_encoded_data = []
    for protein_sequence, distance_matrix, valid_entries in tqdm(data, desc='One-hot encoding', unit='protein'):
        one_hot = torch.zeros(len(protein_sequence), num_unique_amino_acids, dtype=torch.float32).to(device=DEVICE)
        for i, aa in enumerate(protein_sequence):
            one_hot[i, aa_to_index[aa]] = 1
        one_hot = one_hot.view(-1)
        one_hot_encoded_data.append((one_hot, torch.tensor(distance_matrix, dtype=torch.float32).to(
            device=DEVICE), torch.tensor(valid_entries, dtype=torch.float32).to(device=DEVICE)))

    # Now, for each pair of amino acids, concatinate the onehot encoded vectors to create a feature vector
    # ending up with a tensor of shape L x L x 2 * num_unique_amino_acids

    full_matrix_data = []

    for one_hot_encoded, distance_matrix, valid_entries in tqdm(one_hot_encoded_data, desc='Feature matrix', unit='protein'):
        # Calculate protein length based on one_hot_encoded vector size.
        L = one_hot_encoded.numel() // num_unique_amino_acids
        # Reshape flat one-hot vector into (L, num_unique_amino_acids)
        one_hot_matrix = one_hot_encoded.view(L, num_unique_amino_acids)
        # Create the feature matrix using broadcasting: (L, L, num_unique_amino_acids)
        A = one_hot_matrix.unsqueeze(1).expand(L, L, num_unique_amino_acids)
        B = one_hot_matrix.unsqueeze(0).expand(L, L, num_unique_amino_acids)
        feature_matrix = torch.cat((A, B), dim=2)  # shape: (L, L, 2*num_unique_amino_acids)
        # Permute to get channels first: (2*num_unique_amino_acids, L, L)
        feature_matrix = feature_matrix.permute(2, 0, 1)

        full_matrix_data.append((feature_matrix, distance_matrix, valid_entries))

    data = full_matrix_data

    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    dataset = ProteinDataset(train_data)
    validation_dataset = ProteinDataset(test_data)

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = Model(
        input_channels=2 * num_unique_amino_acids,
    ).to(device=DEVICE)
    criterion = MaskedMSELoss()
    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

    train(model, train_dataloader, validation_dataloader, criterion, optimizer, SCALER_GRAD)
