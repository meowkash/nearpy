# A collection of classification models that can be used for facial gesture recognition
# InceptionTime, ROCKET, 1-NN (with DTW) are obvious choices
# Try sktime classifiers 
# AE Feature Extraction
import torch.nn as nn
import torch.nn.functional as F

# Define the Autoencoder model
class TimeSeriesAutoencoder(nn.Module):
    # We will experiment with different encoder sizes 
    def __init__(self, input_size, encoding_size=10):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        self.encoded = self.encoder(x)
        decoded = self.decoder(self.encoded)
        return decoded

class AEWrapper(nn.Module):
    def __init__(self, input_size, encoding_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = TimeSeriesAutoencoder(input_size=input_size, encoding_size=encoding_size)

    def forward(self, x):
        _ = self.model(x) # Forward pass 
        return self.model.encoded

# Define the CNN model
class CWTClassifier(nn.Module):
    def __init__(self, num_channels=16, num_classes=10):
        super(CWTClassifier, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # Adjust for input size
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Apply first convolution, then pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolution, then pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolution (no pooling)
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 4 * 4)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x