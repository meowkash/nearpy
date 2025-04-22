import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import pywt
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import lightning as L
import seaborn as sns

class CWTDataModule(L.LightningDataModule):
    """
    DataModule specifically for handling CWT Scalogram datasets for audio classification.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        wavelet: str = 'morl',
        scales: Optional[np.ndarray] = None,
        num_scales: int = 128,
        normalize: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42
    ):
        """
        Initialize the CWTDataModule.
        
        Args:
            dataframe: Pandas DataFrame with 'RF' and 'Class' columns
            wavelet: Wavelet to use for CWT
            scales: Scales for CWT, if None, will be generated based on num_scales
            num_scales: Number of scales to use if scales is None
            normalize: Whether to normalize the scalograms
            batch_size: Batch size for training/validation/testing
            num_workers: Number of workers for data loading
            train_val_test_split: Proportions for train, validation, and test splits
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.dataframe = dataframe
        self.wavelet = wavelet
        self.scales = scales
        self.num_scales = num_scales
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        
        # Validate split proportions
        assert sum(train_val_test_split) == 1.0, "Split proportions must sum to 1"
        
        # Set up random seed
        pl.seed_everything(seed)
        
    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for the different stages.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or None
        """
        # Create the full dataset
        full_dataset = CWTDataset(
            dataframe=self.dataframe,
            wavelet=self.wavelet,
            scales=self.scales,
            num_scales=self.num_scales,
            normalize=self.normalize,
            augment=False
        )
        
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
            self.train_dataset = CWTDataset(
                dataframe=train_df,
                wavelet=self.wavelet,
                scales=self.scales,
                num_scales=self.num_scales,
                normalize=self.normalize,
                augment=True
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
    
    def get_class_counts(self) -> Dict[str, int]:
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
            plt.title(f'CWT Scalogram Sample {i+1} (Class: {label})')
            plt.xlabel('Time')
            plt.ylabel('Scale')
            plt.tight_layout()
            plt.show()
            
    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of input scalograms (channels, height, width).
        
        Returns:
            Tuple of (channels, height, width)
        """
        # Get a sample
        sample, _ = self.train_dataset[0]
        return sample.shape
            
    def get_optimal_wavelet_parameters(self) -> Dict[str, Any]:
        """
        Get the optimal wavelet parameters based on the dataset.
        
        Returns:
            Dictionary with optimal wavelet parameters
        """
        # This is a placeholder method that would ideally analyze the dataset
        # and determine optimal wavelet parameters. In a real implementation,
        # this could use grid search or other optimization techniques.
        
        return {
            'wavelet': self.wavelet,
            'num_scales': self.num_scales,
            'min_scale': self.scales.min() if self.scales is not None else None,
            'max_scale': self.scales.max() if self.scales is not None else None,
        }
        
class CWTDataset(Dataset):
    """
    Dataset for converting audio data from a DataFrame to CWT scalograms for CNN classification.
    
    Based on literature demonstrating effectiveness of wavelets for audio classification
    (Daubechies et al., 2016; Tzanetakis & Cook, 2002).
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        wavelet: str = 'morl',  # Morlet wavelet is common for audio
        scales: Optional[np.ndarray] = None,
        num_scales: int = 128,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize the CWTDataset.
        
        Args:
            dataframe: Pandas DataFrame containing 'RF' (audio data) and 'Class' (labels) columns
            wavelet: Wavelet to use for CWT
            scales: Scales for CWT, if None, will be generated based on num_scales
            num_scales: Number of scales to use if scales is None
            normalize: Whether to normalize the scalograms
            augment: Whether to apply data augmentation
        """
        self.dataframe = dataframe
        self.wavelet = wavelet
        self.normalize = normalize
        self.augment = augment
        
        # Set scales for CWT
        if scales is None:
            # Generate logarithmically spaced scales
            self.scales = np.logspace(1, 3, num=num_scales)
        else:
            self.scales = scales
            
        # Convert class labels to numerical values if they are strings
        if isinstance(self.dataframe['Class'].iloc[0], str):
            self.classes = self.dataframe['Class'].unique()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.labels = [self.class_to_idx[cls] for cls in self.dataframe['Class']]
        else:
            self.labels = self.dataframe['Class'].values
            
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get the CWT scalogram and label for a given index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (cwt_scalogram, label)
        """
        # Get the audio data and label
        audio_data = self.dataframe['RF'].iloc[idx]
        label = self.labels[idx]
        
        # Convert audio_data to numpy array if it's not already
        if isinstance(audio_data, (list, tuple)):
            audio_data = np.array(audio_data)
        
        # Apply data augmentation if enabled
        if self.augment:
            audio_data = self._augment_audio(audio_data)
        
        # Generate CWT scalogram
        scalogram = self._generate_cwt_scalogram(audio_data)
        
        # Convert to tensor
        scalogram_tensor = torch.from_numpy(scalogram).float()
        
        # Add channel dimension for CNN (1, height, width)
        scalogram_tensor = scalogram_tensor.unsqueeze(0)
        
        return scalogram_tensor, label
    
    def _generate_cwt_scalogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Generate a CWT scalogram from audio data.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            CWT scalogram as a 2D numpy array
        """
        # Generate CWT coefficients
        coeffs, _ = pywt.cwt(audio_data, self.scales, self.wavelet)
        
        # Convert to magnitude (absolute value)
        scalogram = np.abs(coeffs)
        
        # Apply log scaling to enhance visibility of patterns (common in audio)
        scalogram = np.log1p(scalogram)
        
        # Normalize if enabled
        if self.normalize:
            scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())
            
        return scalogram
    
    def _augment_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Augmented audio data
        """
        # Randomly apply time shifting
        if np.random.random() > 0.5:
            shift_amount = int(np.random.random() * len(audio_data) * 0.1)  # Shift up to 10%
            audio_data = np.roll(audio_data, shift_amount)
        
        # Randomly apply additive noise
        if np.random.random() > 0.5:
            noise_level = 0.005 + 0.01 * np.random.random()
            noise = noise_level * np.random.randn(len(audio_data))
            audio_data = audio_data + noise
            
        # Randomly apply amplitude scaling
        if np.random.random() > 0.5:
            scale_factor = 0.8 + np.random.random() * 0.4  # 0.8 to 1.2
            audio_data = audio_data * scale_factor
            
        # Make sure audio length stays consistent
        if len(audio_data) > self.dataframe['RF'].iloc[0].shape[0]:
            audio_data = audio_data[:self.dataframe['RF'].iloc[0].shape[0]]
        elif len(audio_data) < self.dataframe['RF'].iloc[0].shape[0]:
            # Pad with zeros if the audio is too short
            padding = np.zeros(self.dataframe['RF'].iloc[0].shape[0] - len(audio_data))
            audio_data = np.concatenate((audio_data, padding))
            
        return audio_data
    
    def visualize_sample(self, idx: int) -> None:
        """
        Visualize a sample CWT scalogram.
        
        Args:
            idx: Index of the sample to visualize
        """
        scalogram, label = self[idx]
        scalogram = scalogram.squeeze(0).numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(scalogram, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title(f'CWT Scalogram (Class: {label})')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.tight_layout()
        plt.show()