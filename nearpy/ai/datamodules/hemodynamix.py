import pytorch_lightning as pl 
import pandas as pd
from pathlib import Path 
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class HemodynamixDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.inputs = X
        self.target = y
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Load element and transform
        elem = self.inputs[idx]
        if self.transform is not None:
            elem = self.transform(elem)
            # tsai transforms to use: TSVerticalFlip, TSRandomShift, TSHorizontalFlip, TSRandomTrends, TSWarp

        target = self.target[idx]

        return elem, target

class HemodynamixDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir: Path, 
            input_cols: list[str], 
            target_col: str, 
            val_split: float = 0.25, 
            test_split: float = 0.1,
            batch_size: int = 64,
            num_workers: int = 4,
        ):
        
        super().__init__()
        
        # Ensure data dir is always a pathlib.Path object
        assert data_dir.split('.')[-1] == 'pkl', 'data_dir must be a .pkl (pickle) file'
        self.data_dir = data_dir
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split 
        self.test_split = test_split

        self.input_cols = input_cols
        self.target_col = target_col

    def prepare_data(self):
        # Load pickle dataframe and split into train/val/test
        self.dataframe = pd.read_pickle(self.data_dir)

    def setup(self, stage=None):
        X, y = self.dataframe[self.input_cols], self.dataframe[self.target_col]
        
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=self.val_split)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = HemodynamixDataset(X_train, y_train)
            self.val_dataset = HemodynamixDataset(X_val, y_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = HemodynamixDataset(X_test, y_test)

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