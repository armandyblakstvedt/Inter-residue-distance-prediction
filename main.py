from configurations import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE, NUMBER_OF_BATCHES_PER_EPOCH, OPTIMIZER, SCALER_GRAD
from utils.load_data import load_cached_data
from ProteinDataset import ProteinDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.losses import MaskedMSELoss
from train import train
from model import Model
from torch import nn
import torch

if __name__ == '__main__':
    # Load data
    data, dimension = load_cached_data(DEVICE)

    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    # train_data = train_data[:5]
    # test_data = test_data[:1]

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

    # # Load model from file model.pth
    # model = Model(
    #     input_channels=2 * 86,
    # )

    # state_dict = torch.load("model.pth", map_location='cpu')
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict)

    # model.to(DEVICE)

    # # Disable inter pactive mode for saving a static plot.
    # plt.ioff()

    # # Get a random sample from the validation set.
    # sample_data, sample_target, sample_valid = next(iter(validation_dataloader))
    # with torch.no_grad():
    #     sample_pred = model(sample_data)

    # # Convert tensors to numpy arrays.
    # pred_matrix = sample_pred[0, 0].cpu().numpy()
    # target_matrix = sample_target[0].cpu().numpy()

    # # Create a new figure.
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].set_title("Prediction")
    # axs[0].imshow(pred_matrix, cmap='viridis')
    # axs[1].set_title("Target")
    # axs[1].imshow(target_matrix, cmap='viridis')

    # # Save the plot to an image file.
    # fig.savefig(f"plots/a_good_model_v2_{len(sample_data[0])}.png")
    # print("Plot saved as 'validation_sample_plot.png'")
