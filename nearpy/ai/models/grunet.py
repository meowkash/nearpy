import torch
import torchmetrics
from torch import nn 
import lightning as L 

class ConvBNReLU(nn.Module):
    """Simple Conv1d block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GRUNet(L.LightningModule):
    """Simplified encoder-decoder for time series prediction"""
    def __init__(self, 
                 input_channels=1,
                 output_sequence_length=32,
                 hidden_dim=128,
                 num_layers=3,
                 bidirectional=True,
                 loss_fn=None, 
                 dropout=0.1,
                 learning_rate=1e-3,
                 weight_decay=1e-5):
        """
        Args:
            input_channels: Number of input channels
            output_sequence_length: Length of output sequence
            hidden_dim: Hidden dimension for LSTM and Conv layers
            num_layers: Number of LSTM layers
            bidirectional: Whether LSTM is bidirectional
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_channels = input_channels
        self.output_sequence_length = output_sequence_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if loss_fn is None: 
            self.loss_fn = nn.MSELoss()
        else: 
            self.loss_fn = loss_fn
            
        # Feature extraction with convolutional layers
        self.encoder_conv = nn.Sequential(
            ConvBNReLU(input_channels, hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)  # ceil_mode handles odd sizes
        )
        
        # Second level convolutional features
        self.encoder_conv2 = nn.Sequential(
            ConvBNReLU(hidden_dim, hidden_dim * 2),
            ConvBNReLU(hidden_dim * 2, hidden_dim * 2),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)  # ceil_mode handles odd sizes
        )
        
        # LSTM encoder
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Input from conv features
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for sequence processing
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * self.num_directions,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Project encoder output to decoder input dimensions
        self.encoder_to_decoder = nn.Linear(
            hidden_dim * self.num_directions, 
            hidden_dim  # Match decoder hidden dimension
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=1,  # One input at a time for autoregressive prediction
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()
        
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, channels, seq_len]
        Returns:
            Predicted output of shape [batch_size, output_seq_len]
        """
        batch_size = x.size(0)
    
        # Extract features with convolutional layers
        # This handles arbitrary sequence lengths
        conv_features = self.encoder_conv(x)
    
        conv_features = self.encoder_conv2(conv_features)
    
        # Prepare for LSTM: [batch_size, channels, seq_len] -> [batch_size, seq_len, channels]
        lstm_input = conv_features.permute(0, 2, 1)
    
        # Apply LSTM encoder
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)
    
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Get context vector (last output or attention-weighted sum)
        # Using attention-weighted mean
        context = attn_output.mean(dim=1)
        
         # Process hidden states for decoder
        decoder_h = []
        decoder_c = []
        
        # Process each layer's hidden state
        for layer in range(self.num_layers):
            if self.bidirectional:
                # Extract the layer's hidden state and handle bidirectionality
                # h_n shape: [num_layers * num_directions, batch_size, hidden_dim]
                idx = layer * 2  # For bidirectional, each layer has 2 directions
                forward_h = h_n[idx]
                backward_h = h_n[idx + 1]
                
                # Concatenate and project to decoder dimensions
                combined_h = torch.cat([forward_h, backward_h], dim=1)  # [batch_size, hidden_dim*2]
                decoder_h_n = self.encoder_to_decoder(combined_h).unsqueeze(0)  # [1, batch_size, hidden_dim]
                
                # Same for cell state
                forward_c = c_n[idx]
                backward_c = c_n[idx + 1]
                combined_c = torch.cat([forward_c, backward_c], dim=1)
                decoder_c_n = self.encoder_to_decoder(combined_c).unsqueeze(0)
            else:
                # For unidirectional, just take the layer's hidden state
                decoder_h_n = h_n[layer].unsqueeze(0)  # [1, batch_size, hidden_dim]
                decoder_c_n = c_n[layer].unsqueeze(0)
            
            decoder_h.append(decoder_h_n)
            decoder_c.append(decoder_c_n)
        
        # Stack the processed hidden states
        decoder_h_0 = torch.cat(decoder_h, dim=0)  # [num_layers, batch_size, hidden_dim]
        decoder_c_0 = torch.cat(decoder_c, dim=0)  # [num_layers, batch_size, hidden_dim]
        
        # Initialize decoder input with zeros
        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device)
        
        # Generate output sequence
        outputs = []
        
        for i in range(self.output_sequence_length):
            # Run through LSTM
            decoder_output, (decoder_h_0, decoder_c_0) = self.decoder_lstm(
                decoder_input, (decoder_h_0, decoder_c_0)
            )
            
            # Project to output dimension
            prediction = self.output_fc(decoder_output.squeeze(1))
            
            # Store output
            outputs.append(prediction)
            
            # Use current prediction as next input
            decoder_input = prediction.unsqueeze(1)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # [batch_size, output_seq_len]
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        
        # Calculate loss
        # loss = F.mse_loss(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.train_mse(y_hat, y)
        self.train_mae(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', self.train_mse, prog_bar=True)
        self.log('train_mae', self.train_mae, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)

        # Calculate loss
        # loss = F.mse_loss(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', self.val_mse, prog_bar=True)
        self.log('val_mae', self.val_mae, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        
        # Calculate loss
        # loss = F.mse_loss(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.test_mse(y_hat, y)
        self.test_mae(y_hat, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mse', self.test_mse, prog_bar=True)
        self.log('test_mae', self.test_mae, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }