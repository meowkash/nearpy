# Time-Series Auto-encoder model
import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl 

class TimeAutoEncoder(pl.LightningDataModule): 
    # Model Architecture 
    def __init__(self, input_size, encoding_size, optimizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer 
        # Encoder Layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size),
            nn.BatchNorm1d(encoding_size), 
            nn.ReLU()
        )
        # Decoder Layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, input_size),
            nn.BatchNorm1d(input_size), 
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.encoded = self.encoder(x)
        decoded = self.decoder(self.encoded)
        return decoded
    
    # Training Backend
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor: 
        x, y = batch 
        pred = self(x)
        loss = F.mse_loss(pred, y)
        return loss 
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor:
        x, y = batch 
        pred = self(x)
        loss = F.mse_loss(pred, y)
        return loss 
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor: 
        pass
    
    # Configure Backend
    def configure_optimizers(self) -> torch.optim.Optimizer: 
        # By default, Adam is a good choice 
        if self.optimizer is None:         
            return torch.optim.Adam(self.parameters(), lr=0.01)    
        else: 
            return self.optimizer
        
# AE Feature Extraction
# class AEWrapper(nn.Module):
#     def __init__(self, input_size, encoding_size, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.model = TimeSeriesAutoencoder(input_size=input_size, encoding_size=encoding_size)

#     def forward(self, x):
#         _ = self.model(x) # Forward pass 
#         return self.model.encoded