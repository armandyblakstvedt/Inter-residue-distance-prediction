import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):
        # Create a mask of non-NaN values in the target
        mask = ~torch.isnan(target)
        # If no valid target numbers found, return zero loss to avoid division by zero
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        # Calculate squared error only on valid values
        diff = input[mask] - target[mask]
        loss = torch.mean(diff ** 2)
        return loss
