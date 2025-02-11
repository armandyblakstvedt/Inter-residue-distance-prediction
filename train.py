import torch
from tqdm import tqdm
from torch.amp import autocast


def train(model, train_dataloader, val_dataloader, criterion, optimizer, scaler_grad, scheduler, epochs, number_of_batches_per_epoch=None, DEVICE, early_stopping_patience=5):
    model.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    training_losses = []

    progress_bar = tqdm(range(epochs), desc='Training', unit='epoch')
    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        num_train_batches = 0

        # Training loop (scaler_grad applied on training loss)
        for batch_idx, (data, target, valid_entries) in enumerate(train_dataloader):
            if number_of_batches_per_epoch is not None and batch_idx >= number_of_batches_per_epoch:
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
        training_losses.append(train_loss)

        # Validation loop after each epoch
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch_idx, (data, target, valid_entries) in enumerate(val_dataloader):
                if number_of_batches_per_epoch is not None and batch_idx >= number_of_batches_per_epoch:
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

        progress_bar.set_description(f"Training | loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Check for improvement for early stopping (using validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            # Save model
            torch.save(model.state_dict(), "model.pth")
            print("Model saved.")
            break

        # Step scheduler based on validation loss
        scheduler.step(val_loss)
