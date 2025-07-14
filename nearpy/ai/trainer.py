from pathlib import Path
import numpy as np 

from .callbacks import VisualizePredictions 

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

def train_and_evaluate(
        model, 
        config: dict, 
        datamodule: L.LightningDataModule,
        save_checkpoints: bool = True, 
        plot_interval: int = 5, 
        plot_indices: list[int] = None,
        num_samples: int = 5,
        enable_early_stopping: bool = False
):
    # Reproducibility 
    L.seed_everything(42, workers=True)

    # Ensure we have a valid configuration
    assert config is not None, 'Configuration must not be empty'
    
    # Get path to save checkpoints
    base_path = config.get('base_path', Path.cwd())    

    # Get experiment name for saving
    exp_name = config.get('exp_name', 'default')
                                              
    # Define callbacks 
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor_callback]

    if save_checkpoints: 
        ckpt_path = base_path / 'ckpts'
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path, filename=exp_name, save_top_k=1, verbose=False, 
            monitor='val_loss', mode='min', enable_version_counter=True
        )
        callbacks.append(checkpoint_callback)

    if plot_interval is not None: 
        if plot_indices is None: 
            max_idx = len(datamodule.test_dataset) - 1 
            plot_indices = np.random.randint(0, max_idx, size=num_samples)

        plot_callback = VisualizePredictions(
            plot_interval=plot_interval,
            data_indices=plot_indices,
            num_samples=num_samples
        )
        callbacks.append(plot_callback)
    
    if enable_early_stopping: 
        early_callback = EarlyStopping(monitor='val_loss') 
        callbacks.append(early_callback)
    
    logger  = TensorBoardLogger(base_path/'logs', name=exp_name)
    
    # Define trainer 
    trainer = L.Trainer(
        accelerator=config.get('accelerator', 'auto'), 
        devices=config.get('devices', 'auto'), 
        max_epochs=config.get('max_epochs', 100), 
        deterministic=True, 
        check_val_every_n_epoch=1, 
        enable_progress_bar=True, # Uses TQDM Progress Bar by default
        callbacks=callbacks, 
        log_every_n_steps=1, 
        logger=logger
    )

    # Fit model 
    trainer.fit(model, datamodule)
    
    # (Optional) Test model 
    if config.get('test', False): 
        trainer.test(model, datamodule)
    
    # Return everything to visualize stats later 
    return {
        'model': model, 
        'datamodule': datamodule, 
        'trainer': trainer
    }