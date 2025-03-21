from pathlib import Path
import numpy as np 
import seaborn as sns 

import torch 
from torch import nn, optim
import lightning as L
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# DEPRECATION WARNING: THIS IS NO LONGER USED 
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

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    
    #  (Optional) Enable for early stopping
    early_callback = EarlyStopping(monitor='val_loss') 
    
    plot_callback = VisualizePredictions()
    
    logger  = TensorBoardLogger(base_path/'logs', name=exp_name)
    
    # Define trainer 
    trainer = L.Trainer(
        accelerator=config.get('accelerator', 'auto'), 
        devices=config.get('devices', 'auto'), 
        max_epochs=config.get('max_epochs', 100), 
        deterministic=True, 
        check_val_every_n_epoch=1, 
        enable_progress_bar=True, # Uses TQDM Progress Bar by default
        callbacks=[
            checkpoint_callback,
            lr_monitor_callback,
            early_callback, 
            plot_callback
        ], 
        log_every_n_steps=1, 
        logger=logger
    )

    # Fit model 
    trainer.fit(model, datamodule)
    
    # (Optional) Test model 
    if config.get('test', False): 
        trainer.test(model, datamodule)
    
    # Analyze learning rates using Tensorboard
    return model, trainer


class VisualizePredictions(Callback):
    """
    Lightning callback to visualize random test samples at specified epoch intervals.
    """
    def __init__(
        self,
        plot_interval: int = 5,
        num_samples: int = 4,
        figsize: tuple[int, int] = (6, 8),
        data_indices: list[int] = None,
    ):
        """
        Args:
            plot_interval: Visualize predictions every N epochs
            num_samples: Number of random samples to visualize
            figsize: Figure size
            random_seed: For reproducible sampling
            plot_fn: Custom plotting function
            data_indices: Specific indices to visualize instead of random samples
        """
        super().__init__()
        self.plot_interval = plot_interval
        self.num_samples = num_samples
        self.figsize = figsize
        self.data_indices = data_indices
        
    def on_validation_epoch_end(self, trainer, module):
        """ Run visualization at the end of validation epochs.
        trainer: Lightning trainer
        module: LightningModule or LightningDataModule
        """
        
        epoch = trainer.current_epoch
        
        if epoch % self.plot_interval != 0:
            return
        
        dataloader = trainer.datamodule.test_dataloader()
        device = module.device
        
        # Get sample indices
        if self.data_indices is not None:
            indices = self.data_indices
        else:
            max_idx = len(dataloader.dataset) - 1
            indices = np.random.randint(0, max_idx, size=self.num_samples)
            
        # Using seaborn for plotting
        _, axes = plt.subplots(self.num_samples, 3, 
                               figsize=self.figsize, 
                               dpi=300)
        colors = sns.color_palette('husl', self.num_samples*3)
        
        sns.set_style('whitegrid')
         
        for i, idx in enumerate(indices):
            sample = dataloader.dataset[idx]
            x, y = (sample[0], sample[1]) if isinstance(sample, tuple) and len(sample) >= 2 else (sample, None)
            
            # Make prediction
            if len(x.shape) <= 2: 
                x_batch = torch.reshape(x, (x.shape[0], 1, -1)).to(device)
            else:
                x_batch = torch.Tensor(x).to(device)
                
            with torch.no_grad():
                module.eval()
                y_pred = module(x_batch)
            
            # Convert tensors to numpy
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.squeeze(0).cpu().numpy()
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if isinstance(y, torch.Tensor) and y is not None:
                y = y.cpu().numpy()
            
            # Plot
            
            # Handle image data (channels-first format)
            if len(x.shape) == 3 and x.shape[0] in [1, 3]:
                x_display = np.transpose(x, (1, 2, 0))
                if x.shape[0] == 1:
                    x_display = x_display.squeeze(-1)
                axes[i, 0].imshow(x_display, cmap='gray' if x.shape[0] == 1 else None)
                axes[i, 0].set_title("Input")

                # If we're in a classification task, show prediction
                if y is not None and not isinstance(y, np.ndarray):
                    pred_class = np.argmax(y_pred) if len(y_pred.shape) > 0 else y_pred
                    ax.set_title(f"True: {y}, Pred: {pred_class}")

            # Handle 1D data (time series, signals)
            elif len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1):
                if len(x.shape) == 2:
                    x = x.squeeze(0)
                
                sns.lineplot(x, 
                             ax=axes[i, 0], 
                             label='Input',
                             linewidth=3, 
                             color=colors[3*i])

                if y is not None and y_pred is not None:
                    if len(y.shape) <= 1 and len(y_pred.shape) <= 1:
                        sns.lineplot(y, 
                                     ax=axes[i, 1], 
                                     label='Truth',
                                     linewidth=3, 
                                     color=colors[3*i+1])
                        sns.lineplot(y_pred, 
                                     ax=axes[i, 2], 
                                     label='Prediction',
                                     linewidth=3, 
                                     color=colors[3*i+2])
            
            # Default fallback
            else:
                ax.text(0.5, 0.5, f"Input: {x.shape}\nOutput: {y_pred.shape}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()