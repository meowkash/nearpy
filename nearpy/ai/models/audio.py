import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, Tuple

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class AudioCNN(L.LightningModule):
    '''
    Multi-variate CNN model for classifying audio data visualized as images (ex: Spectrogram, Scalograms, etc)

    Inputs: 
        - Audio images: (batch_size, num_channels, time_span, frequency_span)
        - Num classes: Number of classes for classification
        - Dropout rate: Dropout rate for regularization

    Output: 
        - Output tensor of shape (batch_size, num_classes)
    '''
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # CNN architecture
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
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
    
    def training_step(self, 
                      batch: tuple[torch.Tensor, torch.Tensor], 
                      batch_idx: int
                    ) -> dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Log to tensorboard
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)
        
        # Store for epoch end
        self.training_step_outputs.append({'loss': loss, 'acc': acc})
        
        return {'loss': loss, 'acc': acc}
    
    def on_train_epoch_end(self) -> None:
        # Calculate epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = np.mean([x['acc'] for x in self.training_step_outputs])
        
        # Log epoch metrics
        self.log('train_epoch_loss', avg_loss, prog_bar=True, logger=True)
        self.log('train_epoch_acc', avg_acc, prog_bar=True, logger=True)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def validation_step(self, 
                        batch: tuple[torch.Tensor, torch.Tensor], 
                        batch_idx: int
                    ) -> dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Log to tensorboard
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        
        # Store for epoch end
        self.validation_step_outputs.append({'loss': loss, 'acc': acc})
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self) -> None:
        # Calculate epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = np.mean([x['acc'] for x in self.validation_step_outputs])
        
        # Log epoch metrics
        self.log('val_epoch_loss', avg_loss, prog_bar=True, logger=True)
        self.log('val_epoch_acc', avg_acc, prog_bar=True, logger=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, 
                  batch: tuple[torch.Tensor, torch.Tensor], 
                  batch_idx: int
                ) -> dict[str, torch.Tensor]:
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
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self) -> None:
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

class AudioRNN(L.LightningModule):
    '''
    Multi-variate RNN model for classifying audio data processed as 2-D feature vectors (ex: MFCC, LFCC, etc)

    Inputs: 
        - Audio images: (batch_size, num_channels, num_frames, num_features)
        - Hidden size: Size of hidden layers
        - Num layers: Number of RNN layers
        - RNN type: Type of RNN ('lstm' or 'gru')
        - Bi-directional: Whether to use bidirectional RNN
        - Num classes: Number of classes for classification
        - Dropout rate: Dropout rate for regularization

    Output: 
        - Output tensor of shape (batch_size, num_classes)
    '''
    
    def __init__(
        self,
        num_features: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_type: str = 'gru',  # 'lstm' or 'gru'
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # RNN layers
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=num_features,
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
        # Calculate epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = np.mean([x['acc'] for x in self.training_step_outputs])
        
        # Log epoch metrics
        self.log('train_epoch_loss', avg_loss, prog_bar=True)
        self.log('train_epoch_acc', avg_acc, prog_bar=True)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Log to tensorboard
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        
        # Store for epoch end
        self.validation_step_outputs.append({'loss': loss, 'acc': acc})
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self) -> None:
        # Calculate epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = np.mean([x['acc'] for x in self.validation_step_outputs])
        
        # Log epoch metrics
        self.log('val_epoch_loss', avg_loss, prog_bar=True, logger=True)
        self.log('val_epoch_acc', avg_acc, prog_bar=True, logger=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self) -> None:
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