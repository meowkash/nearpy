import torch
from torch import nn         

class DifferentialDetailLoss(nn.Module):
    def __init__(self, 
                 dim: int = 1):
        super(DifferentialDetailLoss, self).__init__()
        self.dim = dim

    def forward(self, inputs, targets):
        # Pad to ensure diff length is the same as target tensor
        diffs = torch.diff(targets, dim=self.dim, 
                           prepend=targets[..., -1].reshape(-1, 1))
        loss = torch.abs(diffs * (targets - inputs))

        return torch.mean(loss)