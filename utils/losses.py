import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self, max_distance=None):
        self.max_distance = max_distance
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):
        B, _, H, W = input.shape
        n = 400
        if H < n or W < n:
            raise ValueError("Input tensor dimensions must be at least 400x400.")

        # Extract a 400x400 submatrix from the top-left of the prediction
        pred_sub = input[:, 0, :n, :n]  # shape: [B, 400, 400]

        # Reshape the target tensor to [B, 400, 400]
        target_matrix = target.view(B, n, n)

        # Create a mask of non-NaN values in the target matrix
        mask = ~torch.isnan(target_matrix)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        
        # If max_distance is provided, only include predicted distances less than max_distance in the loss
        if self.max_distance is not None:
            mask = mask & (pred_sub < self.max_distance)

        # Calculate squared error only on valid values and take the mean
        diff = (pred_sub - target_matrix)[mask]
        loss = torch.mean(diff ** 2)
        return loss
    
class MaskedMAELoss(nn.Module):
    def __init__(self, max_distance=None):
        self.max_distance = max_distance
        super(MaskedMAELoss, self).__init__()

    def forward(self, input, target):
        B, _, H, W = input.shape
        n = 400
        if H < n or W < n:
            raise ValueError("Input tensor dimensions must be at least 400x400.")

        # Extract a 400x400 submatrix from the top-left of the prediction
        pred_sub = input[:, 0, :n, :n]  # shape: [B, 400, 400]

        # Reshape the target tensor to [B, 400, 400]
        target_matrix = target.view(B, n, n)

        # Create a mask of non-NaN values in the target matrix
        mask = ~torch.isnan(target_matrix)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        
        # If max_distance is provided, only include predicted distances less than max_distance in the loss
        if self.max_distance is not None:
            mask = mask & (pred_sub < self.max_distance)

        # Calculate squared error only on valid values and take the mean
        diff = (pred_sub - target_matrix)[mask]
        loss = torch.mean(torch.abs(diff))
        return loss
