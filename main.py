from configurations import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE, NUMBER_OF_BATCHES_PER_EPOCH, OPTIMIZER, SCALER_GRAD
from utils.load_data import load_cached_data
from ProteinDataset import ProteinDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.losses import MaskedMAELoss, MaskedMSELoss
from train import train
from model import Model
from torch import nn
import torch

if __name__ == '__main__':
    # Load data
    data, dimension = load_cached_data(DEVICE)

    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    dataset = ProteinDataset(train_data, dimension)
    validation_dataset = ProteinDataset(test_data, dimension)

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    # Create model
    model = Model(
        input_channels=2 * dimension,
    )
    model = nn.DataParallel(model)
    model.to(DEVICE)

    criterion = MaskedMSELoss()
    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    train(model, train_dataloader, validation_dataloader, criterion, optimizer, SCALER_GRAD, scheduler, EPOCHS, NUMBER_OF_BATCHES_PER_EPOCH, DEVICE, EARLY_STOPPING_PATIENCE)

    # Save model
    torch.save(model.state_dict(), "model.pth")

    # Load model from file model.pth
    model = Model(
        input_channels=2 * 86,
    )

    state_dict = torch.load("model.pth", map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.to(DEVICE)

    # Disable inter pactive mode for saving a static plot.
    plt.ioff()

    loss_fn = MaskedMAELoss()
    total_loss = 0.0
    num_samples = 0

    # Loop over all validation data and get the loss
    for i, data in enumerate(validation_dataloader):
        sequence, target, valid_entries = data
        sequence = sequence.to(DEVICE)
        target = target.to(DEVICE)
        valid_entries = valid_entries.to(DEVICE)

        # Get the model prediction
        with torch.no_grad():
            prediction = model(sequence)

        # Calculate the loss
        loss = loss_fn(prediction, target)

        total_loss += loss.item()
        num_samples += 1

        print(f"Validation sample {i}: Loss: {loss.item()}")

    # Calculate the average loss
    average_loss = total_loss / num_samples
    print(f"Average validation loss: {average_loss}")
