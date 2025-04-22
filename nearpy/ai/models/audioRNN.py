import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union, List, Callable, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


class AudioRNN(pl.LightningModule):
    """
    RNN model (LSTM/GRU) for audio classification.
    
    Architecture inspired by research on sequential models for audio classification
    (Graves et al., 2013, IEEE Transactions on Pattern Analysis and Machine Intelligence;
     Chung et al., 2014, International Conference on Neural Information Processing Systems).
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_type: str = 'gru',  # 'lstm' or 'gru'
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
        learning_rate: float = 0.001
    ):
        """
        Initialize the RNN model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layers
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('lstm' or 'gru')
            bidirectional: Whether to use bidirectional RNN
            dropout_rate: Dropout rate for regularization
            num_classes: Number of classes for classification
            learning_rate: Learning rate for optimization
        """
        super().__init__()
        self.save_hyperparameters()
        
        # RNN layers
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output size after bidirectional RNN
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_size, rnn_output_size // 2),
            nn.Tanh(),
            nn.Linear(rnn_output_size // 2, 1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(rnn_output_size, rnn_output_size // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(rnn_output_size // 2, num_classes)
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Add feature dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, sequence_length, 1)
        
        # RNN layers
        rnn_out, _ = self.rnn(x)  # (batch_size, sequence_length, hidden_size*2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(rnn_out).squeeze(-1), dim=1)  # (batch_size, sequence_length)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_out).squeeze(1)  # (batch_size, hidden_size*2)
        
        # Fully connected layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training."""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Log to tensorboard
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        # Store for epoch end
        self.training_step_outputs.append({'loss': loss, 'acc': acc})
        
        return {'loss': loss, 'acc': acc}
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch."""
        # Calculate epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = np.mean([x['acc'] for x in self.training_step_outputs])
        
        # Log epoch metrics
        self.log('train_epoch_loss', avg_loss, prog_bar=True)
        self.log('train_epoch_acc', avg_acc, prog_bar=True)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and accuracy
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Log to tensorboard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        # Store for epoch end
        self.validation_step_outputs.append({'loss': loss, 'acc': acc})
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""
        # Calculate epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = np.mean([x['acc'] for x in self.validation_step_outputs])
        
        # Log epoch metrics
        self.log('val_epoch_loss', avg_loss, prog_bar=True)
        self.log('val_epoch_acc', avg_acc, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and accuracy
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Store predictions for metrics calculation
        self.test_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': y
        })
        
        # Log to tensorboard
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch."""
        # Collect all predictions and targets
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu()
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs]).cpu()
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        
        # Calculate metrics
        acc = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Log metrics
        self.log('test_loss', avg_loss)
        self.log('test_acc', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        
        # Print metrics
        print(f"\nTest Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Clear outputs
        self.test_step_outputs.clear()


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    DataModule for time-series audio classification using RNNs.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence_length: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        normalize: bool = True,
        seed: int = 42
    ):
        """
        Initialize the DataModule.
        
        Args:
            dataframe: Pandas DataFrame with 'RF' and 'Class' columns
            sequence_length: Length of sequences to use (if None, use full length)
            batch_size: Batch size for training/validation/testing
            num_workers: Number of workers for data loading
            train_val_test_split: Proportions for train, validation, and test splits
            normalize: Whether to normalize the time series data
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.normalize = normalize
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
        full_dataset = TimeSeriesDataset(
            dataframe=self.dataframe,
            sequence_length=self.sequence_length,
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
            self.train_dataset = TimeSeriesDataset(
                dataframe=train_df,
                sequence_length=self.sequence_length,
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
            plt.figure(figsize=(12, 4))
            plt.plot(sample.numpy())
            plt.title(f'Sample {i+1} (Class: {label})')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
