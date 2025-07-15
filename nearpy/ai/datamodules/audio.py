import torch
import pywt
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio.transforms as T

def get_scalogram(data, scales, wavelet, normalize: bool = True):
    data = np.array(data)
    
    coeffs, _ = pywt.cwt(data, scales, wavelet)
    scalogram = np.log1p(np.abs(coeffs))
    if normalize:
        scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())
    
    return scalogram

def augment_data(data, dataframe, data_key): 
    # Randomly apply time shifting
    if np.random.random() > 0.5:
        shift_amount = int(np.random.random() * len(data) * 0.1)  # Shift up to 10%
        data = np.roll(data, shift_amount)
    
    # Randomly apply additive noise
    if np.random.random() > 0.5:
        noise_level = 0.005 + 0.01 * np.random.random()
        noise = noise_level * np.random.randn(len(data))
        data = data + noise
        
    # Randomly apply amplitude scaling
    if np.random.random() > 0.5:
        scale_factor = 0.8 + np.random.random() * 0.4  # 0.8 to 1.2
        data = data * scale_factor
        
    # Make sure audio length stays consistent
    if len(data) > dataframe[data_key].iloc[0].shape[0]:
        data = data[:dataframe[data_key].iloc[0].shape[0]]
    elif len(data) < dataframe[data_key].iloc[0].shape[0]:
        # Pad with zeros if the audio is too short
        padding = np.zeros(dataframe[data_key].iloc[0].shape[0] - len(data))
        data = np.concatenate((data, padding))
        
    return data
    
class SpectrogramDataset(Dataset): 
    def __init__(
        self, 
        dataframe, 
        sample_rate: int, 
        n_fft: int,
        hop_length: int,
        num_vars: int, 
        win_length: int = None, 
        mel: bool = False,
        n_mels: int = None,
        data_key: str = 'Data',
        label_key: str = 'Class',
        augment: bool = False
    ):
        super().__init__()
        self.dataframe = dataframe
        self.num_vars = num_vars
        self.data_key = data_key
        self.label_key = label_key
        self.augment = augment
        
        if mel: 
            # If true, will compute and return mel spectrogram
            self.transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                win_length=win_length,
                norm="slaney"
            )
        else: 
            self.transform = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length
            )  
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        waveform = self.dataframe.iloc[index][self.data_key] # Shape: (num_vars, num_samples)
        labels = self.dataframe.iloc[index][self.label_key]
         
        waveform = np.squeeze(waveform)
        # We reshape waveform to ensure output is always 3D
        if waveform.ndim == 1: 
            waveform = waveform.reshape((1, -1))
        
        if self.augment: 
            waveform = np.apply_along_axis(augment_data, -1, waveform, self.dataframe, self.data_key)
            
        data = self.transform(waveform) # (num_vars, n_fft//2 + 1 (or n_mels), num_frames)
        
        return torch.from_numpy(data), torch.from_numpy(labels)
       
class ScalogramDataset(Dataset): 
    def __init__(
        self, 
        dataframe,
        wavelet, 
        num_vars: int, 
        num_scales: int = 128, # This is tunable for optimal performance
        data_key: str = 'Data',
        label_key: str = 'Class',
        augment: bool = False
    ):        
        super().__init__()
        
        # Keep track of dataframe and labeling 
        self.dataframe = dataframe
        self.num_vars = num_vars
        self.data_key = data_key
        self.label_key = label_key
        
        # Store wavelet as transform
        self.scales = np.arange(1, num_scales)
        self.wavelet = wavelet 
        self.augment = augment
        
    def __len__(self): 
        return len(self.dataframe)
    
    def __getitem__(self, index):
        waveform = self.dataframe.iloc[index][self.data_key] # Shape: (num_vars, num_samples)
        labels = self.dataframe.iloc[index][self.label_key]
         
        waveform = np.squeeze(waveform)
        # We reshape waveform to ensure output is always 3D
        if waveform.ndim == 1: 
            waveform = waveform.reshape((1, -1))
            
        data = np.apply_along_axis(get_scalogram, 0, waveform, self.scales, self.wavelet, self.normalize)
        # Shape: (num_scales, num_vars, num_samples)
        
        if self.augment: 
            data = np.apply_along_axis(augment_data, -1, data, self.dataframe, self.data_key)
            
        # Rearrange to get each var as its own channel for CNN models    
        data = torch.permute(data, (1, 0, 2))
        
        return torch.from_numpy(data), torch.from_numpy(labels)
    
class CepstralDataset(Dataset): 
    """
    Dataset for converting audio data from a DataFrame to MFCC/LFCC features
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        mel: bool = True,
        n_coeffs: int = 128,
        normalize: bool = True,
        data_key: str = 'Data',
        label_key: str = 'Class',
        augment: bool = False
    ):
        self.dataframe = dataframe
        self.data_key = data_key
        self.label_key = label_key
        
        self.normalize = normalize
        self.augment = augment
        
        if mel: 
            self.transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_coeffs,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": n_coeffs
                }
            )
        else: 
            self.transform = T.LFCC(
                sample_rate=sample_rate,
                n_lfcc=n_coeffs,
                speckwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "power": 2.0  
                },
            )
            
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        waveform = self.dataframe.iloc[index][self.data_key] # Shape: (num_vars, num_samples)
        labels = self.dataframe.iloc[index][self.label_key]
         
        waveform = np.squeeze(waveform)
        # We reshape waveform to ensure output is always 3D
        if waveform.ndim == 1: 
            waveform = waveform.reshape((1, -1))
            
        if self.augment: 
            waveform = np.apply_along_axis(augment_data, -1, waveform, self.dataframe, self.data_key)
            
        # Generate cepstral coefficients
        data = self.transform(waveform) # Shape: (num_scales, num_vars, num_samples)
        
        return torch.from_numpy(data), torch.from_numpy(labels)

DATASET_TYPES = {    
    'spectrogram': SpectrogramDataset,
    'scalogram': ScalogramDataset,
    'cepstral': CepstralDataset
}

class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: Path, 
        dataset_type: str,  # Spectrogram, Scalogram
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
            dataset_class: Dataset class to use 
            dataset_args: Arguments to pass to the dataset class
            batch_size: Batch size for training/validation/testing
            num_workers: Number of workers for data loading
            train_val_test_split: Proportions for train, validation, and test splits
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Ensure data dir is always a pathlib.Path object
        self.dataset_dir = Path(dataset_dir)
        self.dataset_class = DATASET_TYPES[dataset_type]
        self.dataset_args = dataset_args
        self.dataframe = None 
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        
        # Validate split proportions
        assert sum(train_val_test_split) == 1.0, "Split proportions must sum to 1"
        
        # Set up random seed
        L.seed_everything(seed)
    
    def prepare_data(self):
        # Load pickle dataframe and split into train/val/test
        if self.dataset_dir.suffix == '.pkl': 
            self.dataframe = pd.read_pickle(self.dataset_dir)
        elif self.dataset_dir.suffix == '.csv': 
            self.dataframe = pd.read_csv(self.dataset_dir)
        else: 
            self.dataframe = None
        
    def setup(self, 
              stage: str = None):
        """
        Set up the datasets for the different stages.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or None
        """
        if self.dataframe is None: 
            self.prepare_data()
        
        # Create the dataset
        dataset = self.dataset_class(self.dataframe, **self.dataset_args)
        
        # Calculate split sizes
        dataset_size = len(dataset)
        train_size = int(self.train_val_test_split[0] * dataset_size)
        val_size = int(self.train_val_test_split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, 
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