# CNN Based Classification Models for image data 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import lightning as L

class CNNClassifier(L.LightningModule): 
    # Model Architecture 
    def __init__(self, input_channels, num_classes, optimizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer 
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # Adjust for input size
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply conv-relu-pool in sequence
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(-1, 64 * 4 * 4)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    # Training Backend
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor: 
        x, y = batch 
        pred = self(x)
        loss = F.mse_loss(pred, y)
        return loss 
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor:
        x, y = batch 
        pred = self(x)
        loss = F.mse_loss(pred, y)
        return loss 
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor: 
        pass
    
    # Configure Backend
    def configure_optimizers(self) -> torch.optim.Optimizer: 
        # By default, Adam is a good choice 
        if self.optimizer is None:         
            return torch.optim.Adam(self.parameters(), lr=0.01)    
        else: 
            return self.optimizer