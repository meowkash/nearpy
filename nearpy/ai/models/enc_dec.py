# Time series encoder-decoder model for transforming one time-series to another
import torch
import torch.nn.functional as F
import lightning as L

class EncodeDecode(L.LightningModule): 
    # Model Architecture 
    def __init__(self, optimizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss 
            
    # Training Backend
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor: 
        x, y = batch 
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log('train_loss', loss)
        return loss 
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor:
        self.evaluate(batch, 'val')
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor: 
        self.evaluate(batch, 'test')
    
    # Configure Backend
    def configure_optimizers(self) -> torch.optim.Optimizer: 
        # By default, Adam is a good choice 
        if self.optimizer is None:         
            return torch.optim.Adam(self.parameters(), lr=0.01)    
        else: 
            return self.optimizer
        
        # Other optimizer options
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )