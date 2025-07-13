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
        
        dataset = trainer.datamodule.test_dataset
        device = module.device
        
        # Get sample indices
        if self.data_indices is not None:
            indices = self.data_indices
        else:
            max_idx = len(dataset) - 1
            indices = np.random.randint(0, max_idx, size=self.num_samples)
            
        # Using seaborn for plotting
        fig, axes = plt.subplots(len(indices), 3, 
                               figsize=self.figsize, 
                               dpi=300)
        num_colors = len(indices) * 3
        colors = sns.color_palette('husl', num_colors)
        
        sns.set_style('whitegrid')
        
        x, y = dataset[indices]
        x = np.array(x.tolist(), dtype=float)
        x = torch.Tensor(x).to(device) # len(indices) x num_vars * num_samples

        # In case we have univariate data, reshape appropriately 
        if len(x.shape) <= 2: 
            x = torch.reshape(x, (x.shape[0], 1, -1))
            
        # Evaluate                 
        with torch.no_grad():
            module.eval()
            y_pred = module(x)
            
        # Convert tensors to numpy
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.squeeze(0).cpu().numpy()
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor) and y is not None:
            y = y.cpu().numpy()
            
        # We will only plot if outputs are valid 
        if y is None or y_pred is None:
            return 
        
        # Plot 
        for i, _ in enumerate(indices): 
            for j in range(x.shape[1]): 
                sns.lineplot(x[i, j], ax=axes[i, 0], 
                             linewidth=3, 
                             color=colors[(3*i+j)%num_colors])
        
            sns.lineplot(y[i], ax=axes[i, 1], 
                         linewidth=3, 
                         color=colors[(3*i+1)%num_colors])
            sns.lineplot(y_pred[i], ax=axes[i, 2], 
                         linewidth=3, 
                         color=colors[(3*i+2)%num_colors])

        # Display title for top graph 
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