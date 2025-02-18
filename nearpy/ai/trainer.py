# Keep Lightning trainer code in this file for simplicity
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import seaborn as sn 
import pandas as pd 
import os 

class LightRunner(pl.LightningModule):
    def __init__(self, model, num_classes, loss=None, optimizer=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
            # Use nn.functional.mse_loss() for auto-encoder
        else:
            self.loss = loss
        self.optimizer = optimizer

    def training_step(self, batch, _):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat.view(x_hat.shape[0], -1), y.long()) 
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, _):
        # This may be buggy 
        x, y = batch
        x_hat = self.model(x.float())
        val_loss = self.loss(x_hat.view(x_hat.shape[0], -1), y.long())
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
            return self.optimizer

class AERunner(pl.LightningModule):
    def __init__(self, model, num_classes, loss=None, optimizer=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        if loss is None:
            self.loss = nn.MSELoss()
            # Use nn.functional.mse_loss() for auto-encoder
        else:
            self.loss = loss
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat.view(x_hat.shape[0], -1), x) 
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx): 
        x, _ = batch
        x_hat = self.model(x.float())
        val_loss = self.loss(x_hat.view(x_hat.shape[0], -1), x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
            return self.optimizer

def train_and_evaluate(model, train_loader, val_loader, max_epochs=10, 
                       num_classes=9, plot=True, loss=None, task='AE', 
                       name='AE_Feat_5', ver=1):
    if task == 'AE':
        ncsRun = AERunner(model, num_classes, loss)
    else:
        ncsRun = LightRunner(model, num_classes, loss)
    
    trainer = pl.Trainer(max_epochs=max_epochs, 
                    accelerator="auto",
                    logger=CSVLogger(save_dir='logs/', name=name, version=ver),
                    enable_progress_bar=False,
                    callbacks=[LearningRateMonitor(logging_interval="epoch"), 
                               ModelCheckpoint(dirpath=os.path.join(os.getcwd(), 'ckpts'), 
                                               filename=name, save_top_k=1, verbose=False, 
                                               monitor='val_loss', mode='min', 
                                               enable_version_counter=False)]
                    )

    trainer.fit(model=ncsRun, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics['val_loss'] 
    
    # Plot learning curves
    if plot:
        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
        metrics.drop(columns=['step', 'lr-Adam', 'epoch'], inplace=True)
        sn.relplot(data=metrics, kind="line")
    
    return ncsRun, trainer, val_loss 