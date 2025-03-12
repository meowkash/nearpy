import torch
import torchmetrics
from torch import nn 
from torch.nn import functional as F
import lightning as L 
import matplotlib.pyplot as plt 
import numpy as np

class GRUNet(L.LightningModule):
    """Combined UNet and GRU model for time series forecasting"""
    
    def __init__(self, 
                 input_channels: int = 1,
                 output_sequence_length: int = 32,
                 base_filters: int = 64,
                 bilinear: bool = True,
                 gru_hidden_dim: int = 64,
                 gru_layers: int = 2,
                 bidirectional: bool = True,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        """
        Args:
            input_channels: Number of input channels
            output_sequence_length: Length of output sequence
            base_filters: Number of base filters in UNet
            bilinear: Whether to use bilinear upsampling
            gru_hidden_dim: Hidden dimension of GRU
            gru_layers: Number of GRU layers
            bidirectional: Whether GRU is bidirectional
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_channels = input_channels
        self.output_sequence_length = output_sequence_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Store GRU output dimensions for dynamic adapter
        self.gru_output_channels = gru_hidden_dim * (2 if bidirectional else 1)
        
        # GRU Encoder
        self.gru_encoder = GRUEncoder(
            input_dim=input_channels,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_layers,
            bidirectional=bidirectional
        )
        
        # UNet blocks - encoder
        self.inc = DoubleConv(input_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        # GRU adapter (will be instantiated dynamically in forward pass)
        self.gru_adapter = None
        
        # UNet Bridge (using simpler structure to avoid dimension issues)
        self.bridge_conv = DoubleConv(base_filters * 8, base_filters * 16)
        
        # UNet blocks - decoder
        factor = 2 if bilinear else 1
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # Output projection
        self.outc = OutConv(base_filters, 1)
        
        # Decoder for output sequence
        num_directions = 2 if bidirectional else 1
        self.decoder_gru = nn.GRU(
            input_size=1,  # We feed 1 value at a time
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False  # Decoder is not bidirectional
        )
        
        self.decoder_fc = nn.Linear(gru_hidden_dim, 1)
        
        # Initialize metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()
        
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
    
    def forward(self, x):
        # x shape: [batch_size, channels, seq_len]
        batch_size = x.size(0)
        
        # UNet encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # GRU encoder
        gru_out, decoder_hidden = self.gru_encoder(x)
        
        # Reshape GRU output to match UNet feature map dimensions
        # gru_out shape: [batch_size, seq_len, gru_hidden_dim * num_directions]
        gru_out = gru_out.permute(0, 2, 1)  # [batch_size, gru_hidden_dim * num_directions, seq_len]
        
        # Adjust GRU output length to match UNet feature map if needed
        if gru_out.size(2) != x4.size(2):
            gru_out = F.interpolate(gru_out, size=x4.size(2), mode='linear', align_corners=False)
        
        # Create GRU adapter dynamically if needed
        if self.gru_adapter is None or self.gru_adapter.in_channels != gru_out.size(1):
            self.gru_adapter = nn.Conv1d(
                gru_out.size(1),
                self.hparams.base_filters * 8,  # Match UNet bottleneck channels
                kernel_size=1
            ).to(x.device)
        
        # Adapt GRU output channels to match UNet bottleneck
        gru_features = self.gru_adapter(gru_out)
        
        # Add features instead of concatenating to avoid dimension issues
        combined = x4 + gru_features
        
        # Bridge - process combined features
        bridge = self.bridge_conv(combined)
        
        # UNet decoder
        x = self.up1(bridge, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, self.inc(x))
        
        # Final UNet output
        unet_out = self.outc(x)  # [batch_size, 1, seq_len]
        
        # Prepare initial decoder input - use the last value from UNet output
        decoder_input = unet_out[:, :, -1:].permute(0, 2, 1)  # [batch_size, 1, 1]
        
        # Generate output sequence
        outputs = []
        
        for i in range(self.output_sequence_length):
            # Run through GRU
            out, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            
            # Project to output dimension
            out = self.decoder_fc(out)
            
            # Collect output
            outputs.append(out)
            
            # Use current prediction as next input
            decoder_input = out
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # [batch_size, output_seq_len, 1]
        
        # Reshape to expected output format
        outputs = outputs.squeeze(-1)  # [batch_size, output_seq_len]
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.train_mse(y_hat, y)
        self.train_mae(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', self.train_mse, prog_bar=True)
        self.log('train_mae', self.train_mae, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', self.val_mse, prog_bar=True)
        self.log('val_mae', self.val_mae, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.test_mse(y_hat, y)
        self.test_mae(y_hat, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mse', self.test_mse, prog_bar=True)
        self.log('test_mae', self.test_mae, prog_bar=True)
        
        # For the first batch, visualize predictions
        if batch_idx == 0 and self.logger is not None:
            try:
                # Plot the first 3 samples of the batch
                num_samples = min(3, x.size(0))
                fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
                
                # Make axes a list if there's only one sample
                if num_samples == 1:
                    axes = [axes]
                
                for i in range(num_samples):
                    ax = axes[i]
                    
                    # Get the input, prediction, and ground truth for this sample
                    input_seq = x[i, 0].detach().cpu().numpy()
                    pred_seq = y_hat[i].detach().cpu().numpy()
                    true_seq = y[i].detach().cpu().numpy()
                    
                    # Plot
                    time_input = np.arange(len(input_seq))
                    time_output = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
                    
                    ax.plot(time_input, input_seq, 'b-', label='Input (Volume)')
                    ax.plot(time_output, true_seq, 'g-', label='True (Pressure)')
                    ax.plot(time_output, pred_seq, 'r--', label='Predicted (Pressure)')
                    
                    ax.set_title(f'Sample {i+1}')
                    ax.legend()
                    ax.grid(True)
                
                plt.tight_layout()
                
                # Log figure to TensorBoard
                self.logger.experiment.add_figure('test_predictions', fig, self.global_step)
            except Exception as e:
                print(f"Warning: Could not create visualization: {e}")
        
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

class GRUEncoder(nn.Module):
    """GRU-based encoder for time series"""
    
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
    
    def forward(self, x):
        # x shape: [batch_size, channels, seq_len]
        # Reshape for GRU: [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        # Run through GRU
        output, hidden = self.gru(x)
        
        # output shape: [batch_size, seq_len, num_directions * hidden_dim]
        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
        
        # Reshape hidden for decoder
        # For bidirectional, concatenate the last hidden state from both directions
        if self.bidirectional:
            # Reshape to [batch_size, num_layers, num_directions, hidden_dim]
            hidden = hidden.view(self.num_layers, self.num_directions, -1, self.hidden_dim)
            
            # Take the last layer's hidden states and concatenate directions
            last_hidden = hidden[-1]
            last_hidden = last_hidden.transpose(0, 1).contiguous().view(-1, self.num_directions * self.hidden_dim).unsqueeze(0)
            
            # Duplicate for decoder layers
            decoder_hidden = last_hidden.repeat(self.num_layers, 1, 1)
        else:
            decoder_hidden = hidden
        
        # Return output sequence and final hidden state
        return output, decoder_hidden

class DoubleConv(nn.Module):
    """Double convolution block for UNet"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block for UNet"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block for UNet"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use bilinear interpolation or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 dimensions if necessary
        diff = x2.size()[2] - x1.size()[2]
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        
        # Concatenate channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution for UNet"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)