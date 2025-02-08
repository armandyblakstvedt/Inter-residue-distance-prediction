import torch
import torch.nn as nn
import math


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):
        # Assume:
        #   input shape: [B, 1, H, W] with H >= n
        #   target shape: [B, n(n+1)/2]
        B, _, H, W = input.shape
        # Infer n from target shape: n(n+1)/2 = target.shape[1]
        target_elements = target.shape[1]
        n = int((math.sqrt(1 + 8 * target_elements) - 1) / 2)
        if n * (n + 1) // 2 != target_elements:
            raise ValueError("Target tensor shape does not match n(n+1)/2 for any integer n.")

        # Extract a square submatrix of size n from the top-left of the prediction
        pred_sub = input[:, 0, :n, :n]  # shape: [B, n, n]
        # Get indices for upper triangle of an n x n matrix
        triu_idx = torch.triu_indices(n, n)
        # Gather the predictions corresponding to the upper triangle.
        preds = pred_sub[:, triu_idx[0], triu_idx[1]]  # shape: [B, n(n+1)/2]

        # Create a mask of non-NaN values in the target
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Calculate squared error only on valid values
        diff = preds[mask] - target[mask]
        loss = torch.mean(diff ** 2)
        return loss
