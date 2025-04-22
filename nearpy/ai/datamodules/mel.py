import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
 
class MelDataset(Dataset):
    """
    Dataset for converting audio data from a DataFrame to mel spectrograms for CNN classification.
    
    This implementation is based on common audio processing techniques for machine learning
    as described in the literature (Hershey et al., 2017; Pons et al., 2018).
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize the MelDataset.
        
        Args:
            dataframe: Pandas DataFrame containing 'RF' (audio data) and 'Class' (labels) columns
            sr: Sampling rate for audio processing
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
            normalize: Whether to normalize the mel spectrograms
            augment: Whether to apply data augmentation
        """
        self.dataframe = dataframe
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize = normalize
        self.augment = augment
        
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
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get the mel spectrogram and label for a given index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (mel_spectrogram, label)
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
        
        # Generate mel spectrogram
        mel_spec = self._generate_mel_spectrogram(audio_data)
        
        # Convert to tensor
        mel_spec_tensor = torch.from_numpy(mel_spec).float()
        
        # Add channel dimension for CNN (1, height, width)
        mel_spec_tensor = mel_spec_tensor.unsqueeze(0)
        
        return mel_spec_tensor, label
    
    def _generate_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Generate a mel spectrogram from audio data.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Mel spectrogram as a 2D numpy array
        """
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize if enabled
        if self.normalize:
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
        return mel_spec_db
    
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
            shift_amount = int(np.random.random() * self.sr * 0.1)  # Shift up to 10% of a second
            audio_data = np.roll(audio_data, shift_amount)
        
        # Randomly apply time stretching
        if np.random.random() > 0.5:
            stretch_factor = 0.8 + np.random.random() * 0.4  # 0.8 to 1.2
            audio_data = librosa.effects.time_stretch(audio_data, rate=stretch_factor)
            
        # Randomly apply pitch shifting
        if np.random.random() > 0.5:
            pitch_shift = np.random.randint(-2, 3)  # -2 to +2 semitones
            audio_data = librosa.effects.pitch_shift(audio_data, sr=self.sr, n_steps=pitch_shift)
            
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
        Visualize a sample mel spectrogram.
        
        Args:
            idx: Index of the sample to visualize
        """
        mel_spec, label = self[idx]
        mel_spec = mel_spec.squeeze(0).numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram (Class: {label})')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        plt.show()
        

class MelSpecDataModule(L.LightningDataModule):
    """
    DataModule specifically for handling Mel Spectrogram datasets for audio classification.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        normalize: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42
    ):
        """
        Initialize the MelSpecDataModule.
        
        Args:
            dataframe: Pandas DataFrame with 'RF' and 'Class' columns
            sr: Sampling rate for audio processing
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
            normalize: Whether to normalize the mel spectrograms
            batch_size: Batch size for training/validation/testing
            num_workers: Number of workers for data loading
            train_val_test_split: Proportions for train, validation, and test splits
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.dataframe = dataframe
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        
        # Validate split proportions
        assert sum(train_val_test_split) == 1.0, "Split proportions must sum to 1"
        
        # Set up random seed
        pl.seed_everything(seed)
        
    def setup(self, stage: str = None):
        """
        Set up the datasets for the different stages.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or None
        """
        # Create the full dataset
        full_dataset = MelDataset(
            dataframe=self.dataframe,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
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
            self.train_dataset = MelDataset(
                dataframe=train_df,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
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
            plt.title(f'Mel Spectrogram Sample {i+1} (Class: {label})')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Frequency Bands')
            plt.tight_layout()
            plt.show()
            
    def get_input_shape(self) -> tuple[int, int, int]:
        """
        Get the shape of input spectrograms (channels, height, width).
        
        Returns:
            Tuple of (channels, height, width)
        """
        # Get a sample
        sample, _ = self.train_dataset[0]
        return sample.shape