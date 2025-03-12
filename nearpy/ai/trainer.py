from pathlib import Path
from torch import nn, optim
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

class LightRunner(L.LightningModule):
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


def train_and_evaluate(model, 
                       config: dict, 
                       datamodule):
    # Reproducibility 
    L.seed_everything(42, workers=True)

    # Ensure we have a valid configuration
    assert config is not None, 'Configuration must not be empty'
    
    # Get path to save checkpoints
    base_path = config.get('base_path', Path.cwd())    
    ckpt_path = base_path / 'ckpts'

    # Get experiment name for saving
    exp_name = config.get('exp_name', 'default')
                                              
    # Define callbacks 
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path, filename=exp_name, save_top_k=1, verbose=False, 
        monitor='val_loss', mode='min', enable_version_counter=True
    )
    logging_callback = TensorBoardLogger(base_path/'logs', name=exp_name)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    
    #  (Optional) Enable for early stopping
    early_callback = EarlyStopping(monitor='val_loss') 
    
    # Define trainer 
    trainer = L.Trainer(
        accelerator=config.get('accelerator', 'auto'), 
        devices=config.get('devices', 'auto'), 
        max_epochs=config.get('max_epochs', 100), 
        deterministic=True, 
        check_val_every_n_epoch=1, 
        enable_progress_bar=True, # Uses TQDM Progress Bar by default
        # callbacks=[
        #     checkpoint_callback,
        #     logging_callback,
        #     lr_monitor_callback
        # ]
    )

    # Fit model 
    trainer.fit(model, datamodule)
    
    # (Optional) Test model 
    if config.get('test', False): 
        trainer.test(model, datamodule)
    
    # Analyze learning rates using Tensorboard
    return model, trainer