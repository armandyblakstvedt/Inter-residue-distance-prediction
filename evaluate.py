from tqdm import tqdm
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
import seaborn as sns
import random
import os

def print_stats(losses):
    # Calculate the average loss
    average_loss = sum(losses) / len(losses)
    print(f"Average loss: {average_loss:.4f}")

    # Print the lowest loss
    lowest_loss = min(losses)
    print(f"Lowest loss: {lowest_loss:.4f}")

    # Print the highest loss
    highest_loss = max(losses)
    print(f"Highest loss: {highest_loss:.4f}")

    # Print the standard deviation of the losses
    std_loss = torch.std(torch.tensor(losses))
    print(f"Standard deviation of the losses: {std_loss:.4f}")

    # Print the percentage of losses within 2 angstroms
    percentage_within_2 = len([x for x in losses if x < 2]) / len(losses) * 100
    print(f"Percentage of losses within 2 angstroms: {percentage_within_2:.2f}%")

    # Print the percentage of losses within 5 angstroms
    percentage_within_5 = len([x for x in losses if x < 5]) / len(losses) * 100
    print(f"Percentage of losses within 5 angstroms: {percentage_within_5:.2f}%")

    # Print the percentage of losses within 8 angstroms
    percentage_within_8 = len([x for x in losses if x < 8]) / len(losses) * 100
    print(f"Percentage of losses within 8 angstroms: {percentage_within_8:.2f}%")

    # Print the percentage of losses within 10 angstroms
    percentage_within_10 = len([x for x in losses if x < 10]) / len(losses) * 100
    print(f"Percentage of losses within 10 angstroms: {percentage_within_10:.2f}%")

if __name__ == '__main__':
    # Load data
    data, dimension = load_cached_data(DEVICE)

    test_data = data[int(0.8 * len(data)):]

    validation_dataset = ProteinDataset(test_data, dimension)

    # train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

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
    losses = []
    max_distances = []
    mean_distances = []

    best_prediction = None
    best_prediction_target = None
    best_prediction_loss = float('inf')

    worst_prediction = None
    worst_prediction_target = None
    worst_prediction_loss = float('-inf')

    random_prediction = None
    random_prediction_target = None
    random_prediction_loss = 0
    random_index = random.randint(0, len(validation_dataloader) - 1)

    # Loop over all validation data and get the loss
    for i, data in tqdm(enumerate(validation_dataloader), desc="Evaluating", total=len(validation_dataloader)):
        sequence, target, valid_entries = data
        sequence = sequence.to(DEVICE)
        target = target.to(DEVICE)
        valid_entries = valid_entries.to(DEVICE)

        # Get the model prediction
        with torch.no_grad():
            prediction = model(sequence)

        # Calculate the loss
        loss = loss_fn(prediction, target)

        # Set the entries in prediction to None where they are not valid
        valid_until_index = valid_entries.long()
        prediction[:, :, valid_until_index:, :] = torch.nan
        prediction[:, :, :, valid_until_index:] = torch.nan

        # Store the best and worst predictions
        if loss.item() < best_prediction_loss:
            best_prediction_loss = loss.item()
            best_prediction = prediction
            best_prediction_target = target
        if loss.item() > worst_prediction_loss:
            worst_prediction_loss = loss.item()
            worst_prediction = prediction
            worst_prediction_target = target

        # Store a random prediction
        if i == random_index:
            random_prediction_loss = loss.item()
            random_prediction = prediction
            random_prediction_target = target   

        losses.append(loss.item())

        # Calculate the maximum value in the target
        max_distance = torch.max(target[0, :valid_until_index, :valid_until_index])
        max_distances.append(max_distance.item())

        # Calculate the mean value in the target
        mean_distance = torch.mean(target[0, :valid_until_index, :valid_until_index])
        mean_distances.append(mean_distance.item())

    # Print statistics
    print("Statistics for all losses:")
    print_stats(losses)

    # Print statistics for losses where the mean distance is less than 30
    filtered_losses = [loss for loss, mean_distance in zip(losses, mean_distances) if mean_distance < 30]
    print("\nStatistics for losses where the mean distance is less than 30:")
    print_stats(filtered_losses)

    # Save the plot to an image file.
    os.makedirs("results", exist_ok=True)

    # Plot the distribution of the losses smoothly using seaborn (dont show the bars)
    plt.figure(figsize=(10, 6))
    sns.histplot(losses, bins=30, kde=True)
    plt.title("Distribution of Losses")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.xticks(range(0, 31, 1))
    plt.grid()
    plt.savefig("results/loss_distribution.png")
    print("Plot saved as 'loss_distribution.png'")
    plt.show()

    # Plot the best, worst and random prediction in a 3x2 grid
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Best, Worst and Random Prediction")
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs[0, 0].imshow(best_prediction[0, 0].cpu().numpy(), cmap='viridis')
    axs[0, 0].set_title(f"Best Prediction (Error: {best_prediction_loss:.4f} Å)")
    axs[0, 1].imshow(best_prediction_target[0].cpu().numpy(), cmap='viridis')
    axs[0, 1].set_title("Target (Best Prediction)")
    axs[1, 0].imshow(worst_prediction[0, 0].cpu().numpy(), cmap='viridis')
    axs[1, 0].set_title(f"Worst Prediction (Error: {worst_prediction_loss:.4f} Å)")
    axs[1, 1].imshow(worst_prediction_target[0].cpu().numpy(), cmap='viridis')
    axs[1, 1].set_title("Target (Worst Prediction)")
    axs[2, 0].imshow(random_prediction[0, 0].cpu().numpy(), cmap='viridis')
    axs[2, 0].set_title(f"Random Prediction (Error: {random_prediction_loss:.4f} Å)")
    axs[2, 1].imshow(random_prediction_target[0].cpu().numpy(), cmap='viridis')
    axs[2, 1].set_title("Target (Random Prediction)")
    plt.colorbar(axs[0, 0].imshow(best_prediction[0, 0].cpu().numpy(), cmap='viridis'), ax=axs[0, 0])
    plt.colorbar(axs[0, 1].imshow(best_prediction_target[0].cpu().numpy(), cmap='viridis'), ax=axs[0, 1])
    plt.colorbar(axs[1, 0].imshow(worst_prediction[0, 0].cpu().numpy(), cmap='viridis'), ax=axs[1, 0])
    plt.colorbar(axs[1, 1].imshow(worst_prediction_target[0].cpu().numpy(), cmap='viridis'), ax=axs[1, 1])
    plt.colorbar(axs[2, 0].imshow(random_prediction[0, 0].cpu().numpy(), cmap='viridis'), ax=axs[2, 0])
    plt.colorbar(axs[2, 1].imshow(random_prediction_target[0].cpu().numpy(), cmap='viridis'), ax=axs[2, 1])

    plt.savefig("results/best_worst_random_prediction.png")
    print("Plot saved as 'best_worst_random_prediction.png'")

    plt.show()

    # Plot the maximum distance against the loss
    plt.figure(figsize=(10, 6))
    plt.scatter(max_distances, losses, alpha=0.5)
    plt.title("Maximum Distance vs Loss")
    plt.xlabel("Maximum Distance (Å)")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("results/max_distance_vs_loss.png")
    print("Plot saved as 'max_distance_vs_loss.png'")
    plt.show()

    # Plot the mean distance against the loss
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_distances, losses, alpha=0.5)
    plt.title("Mean Distance vs Loss")
    plt.xlabel("Mean Distance (Å)")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("results/mean_distance_vs_loss.png")
    print("Plot saved as 'mean_distance_vs_loss.png'")
    plt.show()

    plt.close()
