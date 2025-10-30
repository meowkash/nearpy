'''
General data-module implementation which provides multivariate time series as inputs 
and a single time series as output. While this may not work for all cases, this can
be a good starting point for custom DataModules
'''

import pandas as pd
from pathlib import Path 
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import lightning as L
import numpy as np 

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.inputs = X
        self.target = y
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Load element and transform
        elem = np.stack(self.inputs[idx]) # This ensures that array is unfolded for multi-variate datasets 
        
        if elem.ndim == 1:
            elem = elem.reshape(1, -1)
            
        if self.transform is not None:
            elem = self.transform(elem)
            # Available transforms may be found in nearpy.ai.augmentations

        target = self.target[idx]
        
        return elem, target

class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_dir: Path, 
            input_cols: list[str], 
            target_col: str, 
            val_split: float = 0.25, 
            test_split: float = 0.1,
            batch_size: int = 64,
            num_workers: int = 4,
            seed: int = 42
        ):
        
        super().__init__()
        
        # Ensure data dir is always a pathlib.Path object
        self.data_dir = data_dir
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split 
        self.test_split = test_split

        self.input_cols = input_cols
        self.target_col = target_col

        L.seed_everything(seed)
        
    def prepare_data(self):
        # Load pickle dataframe and split into train/val/test
        if self.data_dir.suffix == '.pkl': 
            self.dataframe = pd.read_pickle(self.data_dir)
        elif self.data_dir.suffix == '.csv': 
            self.dataframe = pd.read_csv(self.data_dir)
        else: 
            self.dataframe = None

    def setup(self):
        X = self.dataframe[self.input_cols].to_numpy() 
        y = self.dataframe[self.target_col].to_numpy()
        
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=self.val_split)

        # Assign train/val datasets for use in dataloaders
        self.train_dataset = TimeSeriesDataset(X_train, y_train)
        self.val_dataset = TimeSeriesDataset(X_val, y_val)

        # Assign test dataset for use in dataloader(s)
        self.test_dataset = TimeSeriesDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )