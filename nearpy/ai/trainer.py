from pathlib import Path
import numpy as np 

from .callbacks import VisualizePredictions 

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

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
    base_path = Path(config.get('base_path', Path.cwd()))

    # Get experiment name for saving
    exp_name = config.get('exp_name', 'default')
                                              
    # Define callbacks 
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor_callback]

    if save_checkpoints: 
        ckpt_path = base_path / 'Checkpoints'
        ckpt_path.mkdir(exist_ok=True, parents=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path, 
            filename=f"{exp_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            save_top_k=1, 
            verbose=False, 
            monitor='val_loss', 
            mode='min'
        )
        callbacks.append(checkpoint_callback)

    if isinstance(plot_interval, int) and plot_interval > 0: 
        if plot_indices is None: 
            max_idx = len(datamodule.test_dataset) - 1 
            plot_indices = np.random.randint(0, max_idx, size=num_samples)

        visualize_path = base_path / 'Visualization'
        visualize_path.mkdir(exist_ok=True, parents=True)

        plot_callback = VisualizePredictions(
            visualize_path=visualize_path,
            exp_name=exp_name, 
            plot_interval=plot_interval,
            data_indices=plot_indices,
            num_samples=num_samples
        )
        callbacks.append(plot_callback)
    
    if enable_early_stopping: 
        early_callback = EarlyStopping(monitor='val_loss') 
        callbacks.append(early_callback)
    
    csv_logger = CSVLogger(save_dir=base_path / 'logs', name=exp_name, version='csv_logs')
    tb_logger  = TensorBoardLogger(save_dir=base_path/ 'logs', name=exp_name, version='tb_logs')
    
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
        logger=[tb_logger, csv_logger]
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
        'trainer': trainer,
        'best_model': checkpoint_callback.best_model_path
    }