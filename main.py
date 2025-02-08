import torch
import tqdm
from torch.amp import autocast
from utils.load_data import load_data, load_cached_data
from ProteinDataset import ProteinDataset
from torch.utils.data import DataLoader
from model import Model
from utils.losses import MaskedMSELoss
import os
from utils.visualize import visualize_distances
import numpy as np
import matplotlib.pyplot as plt  # New import

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


def train(model, train_dataloader, val_dataloader, criterion, optimizer, scaler_grad):
    model.train()
    # Scheduler based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm.tqdm(range(EPOCHS), desc='Epochs', unit='epoch'):
        model.train()
        running_loss = 0.0
        num_train_batches = 0

        # Training loop (scaler_grad applied on training loss)
        for batch_idx, (data, target) in enumerate(train_dataloader):
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
            for batch_idx, (data, target) in enumerate(val_dataloader):
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

        # --- LIVE UPDATE VISUALIZATION ---
        # Pick one sample from the validation set for live visualization.
        sample_data, sample_target = next(iter(val_dataloader))
        with torch.no_grad():
            with autocast(device_type=DEVICE, dtype=torch.float32):
                sample_output = model(sample_data)
        # Reconstruct the symmetric matrix from the flattened upper triangle (prediction)
        sample_output = sample_output.cpu().numpy().flatten()
        pred_matrix = reconstruct_symmetric(sample_output)

        # Reconstruct the symmetric matrix from the flattened upper triangle (target)
        target_flat = sample_target.cpu().numpy().flatten()
        target_matrix = reconstruct_symmetric(target_flat)

        # Plot predicted and target distance matrices side by side.
        plt.figure("Distance Matrix Comparison")
        plt.clf()  # Clear the current figure
        plt.subplot(1, 2, 1)
        plt.title("Predicted")
        plt.imshow(pred_matrix, cmap='viridis')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("Target")
        plt.imshow(target_matrix, cmap='viridis')
        plt.colorbar()
        plt.pause(0.1)  # Pause to update the figure
        # --- END LIVE UPDATE VISUALIZATION ---

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


def reconstruct_symmetric(upper_triangle, size=400):
    import numpy as np
    upper_triangle = np.array(upper_triangle).flatten()  # Ensure a 1D array

    # Create a full matrix filled with NaN
    M = np.full((size, size), np.nan)

    # Determine the number of valid entries (assumed to be contiguous at the start)
    L_valid = 0
    for val in upper_triangle:
        if np.isnan(val):
            break
        L_valid += 1

    # Calculate n such that n(n+1)/2 equals the number of valid entries.
    n = int((-1 + np.sqrt(1 + 8 * L_valid)) / 2)

    idx = 0
    for i in range(n):
        for j in range(i, n):
            M[i, j] = upper_triangle[idx]
            M[j, i] = upper_triangle[idx]
            idx += 1

    return M


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
    for protein_sequence, distance_matrix, valid_entries in data:
        one_hot = torch.zeros(len(protein_sequence), num_unique_amino_acids, dtype=torch.float32).to(device=DEVICE)
        for i, aa in enumerate(protein_sequence):
            one_hot[i, aa_to_index[aa]] = 1
        one_hot = one_hot.view(-1)
        one_hot_encoded_data.append((one_hot, torch.tensor(distance_matrix, dtype=torch.float32).to(device=DEVICE), valid_entries))

    data = one_hot_encoded_data

    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    dataset = ProteinDataset(train_data)
    validation_dataset = ProteinDataset(test_data)

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = Model().to(device=DEVICE)
    criterion = MaskedMSELoss()
    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

    train(model, train_dataloader, validation_dataloader, criterion, optimizer, SCALER_GRAD)

    # Final evaluation on validation set after training
    validation_loss = 0.0
    num_batches = 0
    validation_predictions = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_dataloader):
            if NUMBER_OF_BATCHES_PER_EPOCH is not None and batch_idx >= NUMBER_OF_BATCHES_PER_EPOCH:
                break
            output = model(data)
            loss = criterion(output, target)
            validation_loss += loss.item()
            num_batches += 1
            validation_predictions.append(output.cpu().numpy())
    if num_batches > 0:
        validation_loss /= num_batches
    else:
        validation_loss = float('nan')
    print(f"Validation Loss: {validation_loss:.4f}")

    validation_predictions = [reconstruct_symmetric(pred) for pred in validation_predictions]
    predictions_dir = os.path.abspath('./predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    save_path = os.path.join(predictions_dir, "validation_predictions.txt")
    with open(save_path, 'w') as f:
        for pred in validation_predictions:
            f.write(str(pred.tolist()) + "\n")

    visualize_distances(validation_predictions)
    # Optionally block interactive plotting at the end
    plt.ioff()
    plt.show()
