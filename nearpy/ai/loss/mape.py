import torch
from torch import nn 

class MAPELoss(nn.Module): 
    def __init__(self, epsilon=1e-6):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Calculate the MAPE loss.

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): Actual values

        Returns:
            torch.Tensor: Mean Absolute Percentage Error
        """
        # Ensure no negative values in y_true
        if torch.any(y_true <= 0):
            raise ValueError("MAPE is undefined for values <= 0 in y_true")

        # Calculate the absolute percentage error
        error = torch.abs((y_true - y_pred) / (y_true + self.epsilon))

        # Remove any infinite values and convert to percentage
        error = torch.where(torch.isinf(error), torch.zeros_like(error), error)

        return torch.mean(error) * 100