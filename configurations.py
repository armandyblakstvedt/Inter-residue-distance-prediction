import torch

# Torch configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Training parameters
EPOCHS = 10
NUMBER_OF_BATCHES_PER_EPOCH = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 100

# Training options
OPTIMIZER = torch.optim.AdamW
SCALER_GRAD = torch.amp.grad_scaler.GradScaler()
