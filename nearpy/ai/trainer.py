from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 

from .callbacks import VisualizePredictions 

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

def train_and_evaluate(model, 
                       config: dict, 
                       datamodule: L.LightningDataModule,
                       plot_interval: int = 5, 
                       plot_indices: list[int] = None,
                       num_samples: int = 5):
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

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    
    if plot_indices is None: 
        max_idx = len(datamodule.test_dataset) - 1 
        plot_indices = np.random.randint(0, max_idx, size=num_samples)
    
    plot_callback = VisualizePredictions(
        plot_interval=plot_interval,
        data_indices=plot_indices,
        num_samples=num_samples
    )
    
    callbacks = [
        checkpoint_callback, 
        lr_monitor_callback,
        plot_callback
    ]
    
    if config.get('early_stopping', False): 
        #  (Optional) Enable for early stopping
        early_callback = EarlyStopping(monitor='val_loss') 
        callbacks.append[early_callback]
    
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
    

# Training utility functions
def plot_training_progress(trainer: L.Trainer, 
                           model: L.LightningModule) -> None:
    """
    Plot the training progress (loss and accuracy).
    
    Args:
        trainer: PyTorch Lightning Trainer
        model: Trained model
    """
    metrics = trainer.callback_metrics
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(model.trainer.logged_metrics.get('train_loss', []), label='Train Loss')
    ax1.plot(model.trainer.logged_metrics.get('val_loss', []), label='Val Loss')
    if 'test_loss' in model.trainer.logged_metrics:
        ax1.plot(model.trainer.logged_metrics.get('test_loss', []), label='Test Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(model.trainer.logged_metrics.get('train_acc', []), label='Train Acc')
    ax2.plot(model.trainer.logged_metrics.get('val_acc', []), label='Val Acc')
    if 'test_acc' in model.trainer.logged_metrics:
        ax2.plot(model.trainer.logged_metrics.get('test_acc', []), label='Test Acc')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()