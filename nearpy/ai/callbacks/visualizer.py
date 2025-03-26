import torch 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

from lightning.pytorch.callbacks import Callback

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
        """ Run visualization at the end of testing epoch. This ensures we have test loss to display
        trainer: Lightning trainer
        module: LightningModule or LightningDataModule
        """
        
        epoch = trainer.current_epoch
        test_loss = trainer.callback_metrics.get('val_loss', 0)
        
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
        fig, axes = plt.subplots(len(indices), 3, 
                               figsize=self.figsize, 
                               dpi=300)
        colors = sns.color_palette('husl', len(indices)*3)
        
        sns.set_style('whitegrid')
         
        for i, idx in enumerate(indices):
            sample = dataloader.dataset[idx]
            x, y = (sample[0], sample[1]) if isinstance(sample, tuple) and len(sample) >= 2 else (sample, None)
            
            x_batch = torch.Tensor(x).to(device)
            
            # Make prediction
            if len(x.shape) <= 2: 
                x_batch = torch.reshape(x_batch, (x_batch.shape[0], 1, -1))
                
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
                    axes[i, 2].set_title(f"True: {y}, Pred: {pred_class}")

            # Handle 1D data (time series, signals)
            elif len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1):
                if len(x.shape) == 2:
                    x = x.squeeze(0)
                
                sns.lineplot(x, 
                             ax=axes[i, 0], 
                             linewidth=3, 
                             color=colors[3*i])
                if y is not None and y_pred is not None:
                    if len(y.shape) <= 1 and len(y_pred.shape) <= 1:
                        sns.lineplot(y, 
                                     ax=axes[i, 1], 
                                     linewidth=3, 
                                     color=colors[3*i+1])
                        sns.lineplot(y_pred, 
                                     ax=axes[i, 2], 
                                     linewidth=3, 
                                     color=colors[3*i+2])
            
                # Display title for top graph 
                if i == 0: 
                    axes[i, 0].set_title('Input')
                    axes[i, 1].set_title('Target')
                    axes[i, 2].set_title('Prediction')
                
                if i!=len(indices): 
                    axes[i, 0].set_xticklabels([])
                    axes[i, 1].set_xticklabels([])
                    axes[i, 2].set_xticklabels([])

        fig.suptitle(f'Epoch {epoch}. Val Loss: {test_loss}')            
        fig.supxlabel('Interpolated Time Axis')
        plt.tight_layout()
        plt.show()