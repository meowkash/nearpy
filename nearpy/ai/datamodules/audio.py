import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

class AudioDataModule(L.LightningDataModule):
    """
    DataModule for audio classification using either Mel spectrograms or CWT scalograms.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_class,  # MelDataset or CWTDataset
        dataset_args: dict[str],
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42
    ):
        """
        Initialize the DataModule.
        
        Args:
            dataframe: Pandas DataFrame with 'RF' and 'Class' columns
            dataset_class: Dataset class to use (MelDataset or CWTDataset)
            dataset_args: Arguments to pass to the dataset class
            batch_size: Batch size for training/validation/testing
            num_workers: Number of workers for data loading
            train_val_test_split: Proportions for train, validation, and test splits
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.dataframe = dataframe
        self.dataset_class = dataset_class
        self.dataset_args = dataset_args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        
        # Validate split proportions
        assert sum(train_val_test_split) == 1.0, "Split proportions must sum to 1"
        
        # Set up random seed
        L.seed_everything(seed)
        
    def setup(self, 
              stage: str = None):
        """
        Set up the datasets for the different stages.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or None
        """
        # Create the full dataset
        full_dataset = self.dataset_class(self.dataframe, **self.dataset_args)
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * dataset_size)
        val_size = int(self.train_val_test_split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # For training, enable augmentation for the training set
        if stage == 'fit' or stage is None:
            # Create a new dataset with augmentation for training
            train_df = self.dataframe.iloc[self.train_dataset.indices]
            self.train_dataset = self.dataset_class(
                train_df,
                **{**self.dataset_args, 'augment': True}
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_counts(self) -> dict[str, int]:
        """
        Get the class distribution in the full dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        return self.dataframe['Class'].value_counts().to_dict()
    
    def plot_class_distribution(self) -> None:
        """Plot the class distribution in the dataset."""
        class_counts = self.dataframe['Class'].value_counts()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def visualize_samples(self, num_samples: int = 5) -> None:
        """
        Visualize random samples from the training set.
        
        Args:
            num_samples: Number of samples to visualize
        """
        for i in range(min(num_samples, len(self.train_dataset))):
            # Get a random index
            idx = np.random.randint(0, len(self.train_dataset))
            
            # Get the sample
            sample, label = self.train_dataset[idx]
            
            # Visualize
            plt.figure(figsize=(10, 4))
            plt.imshow(sample.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar()
            plt.title(f'Sample {i+1} (Class: {label})')
            plt.xlabel('Time')
            plt.ylabel('Frequency/Scale')
            plt.tight_layout()
            plt.show()


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


# Example usage
"""
# Create a DataModule
datamodule = AudioDataModule(
    dataframe=your_dataframe,
    dataset_class=MelDataset,  # or CWTDataset
    dataset_args={
        'sr': 22050,
        'n_fft': 2048,
        'hop_length': 512,
        'n_mels': 128,
        'normalize': True
    },
    batch_size=32,
    train_val_test_split=(0.7, 0.15, 0.15)
)

# Create the model
model = AudioCNN(
    input_channels=1,
    num_classes=len(datamodule.dataframe['Class'].unique()),
    learning_rate=0.001
)

# Create a trainer
trainer = pl.Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)
    ]
)

# Train the model
trainer.fit(model, datamodule)

# Test the model
trainer.test(model, datamodule)

# Plot training progress
plot_training_progress(trainer, model)
"""