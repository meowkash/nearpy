import torch.nn as nn

# Base Model Architectures
class CNNConverter(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        kernel_size: int = 3, 
        output_dim: int = 1
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        # Transpose for conv1d (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        conv_out = self.conv_layers(x)
        
        # Global pooling
        pooled = self.adaptive_pool(conv_out).squeeze(-1)
        
        # Fully connected output
        out = self.fc(pooled)
        return out

class HybridCNNLSTMConverter(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        cnn_hidden: int = 64, 
        lstm_hidden: int = 64, 
        output_dim: int = 1
    ):
        super().__init__()
        
        # CNN Feature Extractor
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, cnn_hidden, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_hidden, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(lstm_hidden, output_dim)
    
    def forward(self, x):
        # Transpose for conv1d
        x_conv = x.transpose(1, 2)
        
        # CNN feature extraction
        cnn_out = self.cnn_layers(x_conv)
        
        # Prepare for LSTM (back to original shape)
        lstm_in = cnn_out.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_in)
        
        # Take last time step
        out = self.fc(lstm_out[:, -1, :])
        return out