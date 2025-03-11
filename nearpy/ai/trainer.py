from pathlib import Path

import seaborn as sn 
import pandas as pd 
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

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


def train_and_evaluate(model, config, train_loader, val_loader, 
                       plot=True, loss=None, 
                       name="", ver=1, base_path=None):
    # Define path to save checkpoints
    if base_path is None: 
        base_path = Path.cwd() 
    ckpt_path = base_path / 'ckpts'
    
    assert config is not None, 'Configuration must not be empty'
    
    if name == "": 
        name = f'default_{task}'
    
    # trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", enable_progress_bar=True, 
    #                                logger=CSVLogger(save_dir='logs/', name=name, version=ver),
    #                                callbacks=[LearningRateMonitor(logging_interval="epoch"), 
    #                                           ModelCheckpoint(dirpath=ckpt_path, filename=name, 
    #                                                           save_top_k=1, verbose=False, monitor='val_loss', 
    #                                                           mode='min', enable_version_counter=False)])
    trainer = pl.Trainer(
        accelerator=config.accelerator, 
        devices=config.devices, 
        max_epochs=config.max_epochs, 
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics['val_loss']
    
    # Plot learning curves
    if plot:
        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
        metrics.drop(columns=['step', 'lr-Adam', 'epoch'], inplace=True)
        sn.relplot(data=metrics, kind="line")
    
    return model, trainer, val_loss