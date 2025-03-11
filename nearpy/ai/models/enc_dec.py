# Time series encoder-decoder model for transforming one time-series to another
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

class EncodeDecode(pl.LightningDataModule): 
    # Model Architecture 
    def __init__(self, optimizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    # Training Backend
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor: 
        x, y = batch 
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log(f'Train Loss: {loss}')
        return loss 
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor:
        x, y = batch 
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log(f'Val Loss: {loss}')
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