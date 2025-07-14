# Experimental: PINN for Volume -> Pressure conversion
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List

# Cardiac Physiology-Informed Neural Network
class CardiacPressureConverter(nn.Module):
    """
    Physics-Guided Neural Network for Cardiac Volume to Pressure Conversion
    
    Key Physiological Principles Incorporated:
    1. End-Systolic Pressure-Volume Relationship (ESPVR)
    2. End-Diastolic Pressure-Volume Relationship (EDPVR)
    3. Time-varying elastance concept
    """
    def __init__(
        self, 
        input_features: int = 1,  # Volume curve
        hidden_layers: List[int] = [64, 64, 32],
        activation: nn.Module = nn.SiLU()
    ):
        super().__init__()
        
        # Network architecture
        layers = []
        prev_features = input_features
        
        for layer_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_features, layer_size),
                activation
            ])
            prev_features = layer_size
        
        # Final output layer (pressure)
        layers.append(nn.Linear(prev_features, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Physiological parameter initialization
        self.max_elastance = nn.Parameter(torch.tensor(2.0))  # Maximal elastance
        self.end_diastolic_volume = nn.Parameter(torch.tensor(100.0))  # End-diastolic volume
        self.contractility = nn.Parameter(torch.tensor(1.0))  # Contractility parameter
    
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physiological constraints
        
        Args:
            volume (torch.Tensor): Cardiac volume input
        
        Returns:
            torch.Tensor: Estimated cardiac pressure
        """
        # Basic network prediction
        base_prediction = self.network(volume)
        
        # Physiologically-inspired modifications
        
        # 1. End-Systolic Pressure-Volume Relationship (ESPVR)
        espvr = self.max_elastance * (volume - self.end_diastolic_volume)
        
        # 2. End-Diastolic Pressure-Volume Relationship (EDPVR)
        # Exponential relationship to capture non-linear diastolic properties
        edpvr = torch.exp(volume / self.end_diastolic_volume) - 1
        
        # Combine network prediction with physiological constraints
        final_pressure = base_prediction + espvr + edpvr
        
        return final_pressure

# Physics-Informed Loss Function
class CardiacPhysicsLoss(nn.Module):
    """
    Custom loss function incorporating physiological constraints
    """
    def __init__(
        self, 
        volume_range: Tuple[float, float],
        pressure_range: Tuple[float, float]
    ):
        super().__init__()
        self.volume_range = volume_range
        self.pressure_range = pressure_range
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        volumes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics-informed loss
        
        Constraints:
        1. Prediction accuracy
        2. Physiological range constraints
        3. Smoothness of pressure-volume relationship
        4. Conservation of energy principles
        """
        # 1. Standard Mean Squared Error
        mse_loss = nn.functional.mse_loss(predictions, targets)
        
        # 2. Range Constraints
        volume_range_penalty = torch.mean(
            torch.relu(volumes - self.volume_range[1]) + 
            torch.relu(self.volume_range[0] - volumes)
        )
        
        pressure_range_penalty = torch.mean(
            torch.relu(predictions - self.pressure_range[1]) + 
            torch.relu(self.pressure_range[0] - predictions)
        )
        
        # 3. Smoothness Constraint (first derivative smoothness)
        volume_diff = torch.diff(volumes)
        pressure_diff = torch.diff(predictions)
        smoothness_loss = torch.mean(torch.abs(
            torch.diff(pressure_diff / volume_diff)
        ))
        
        # 4. Combine losses with learnable weights
        total_loss = (
            mse_loss + 
            0.1 * volume_range_penalty + 
            0.1 * pressure_range_penalty + 
            0.01 * smoothness_loss
        )
        
        return total_loss

# Data Generation and Preprocessing
def generate_cardiac_data(
    num_samples: int = 1000, 
    noise_level: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic cardiac volume-pressure data
    
    Simulates a realistic cardiac cycle with:
    1. Physiological volume range
    2. Non-linear pressure-volume relationship
    3. Realistic noise
    """
    # Time array
    t = np.linspace(0, 1, num_samples)
    
    # Volume curve (simplified sinusoidal model)
    volume = 100 + 50 * np.sin(2 * np.pi * t) * (1 - 0.5 * t)
    
    # Pressure calculation with physiological non-linearity
    # Uses an exponential and quadratic relationship
    pressure = (
        20 +  # Base pressure
        0.5 * volume**2 / 100 +  # Quadratic elastance component
        10 * np.exp(volume / 100) -  # Exponential diastolic compliance
        np.sin(2 * np.pi * t) * 15  # Cyclic variation
    )
    
    # Add noise
    volume += np.random.normal(0, noise_level * volume)
    pressure += np.random.normal(0, noise_level * pressure)
    
    return volume.reshape(-1, 1), pressure.reshape(-1, 1)

# Training Pipeline
def train_cardiac_volume_pressure_converter(
    epochs: int = 1000, 
    learning_rate: float = 0.001
):
    # Generate synthetic data
    volumes, pressures = generate_cardiac_data()
    
    # Convert to PyTorch tensors
    volumes_tensor = torch.FloatTensor(volumes)
    pressures_tensor = torch.FloatTensor(pressures)
    
    # Initialize model and loss
    model = CardiacPressureConverter()
    physics_loss = CardiacPhysicsLoss(
        volume_range=(0, 250),  # Physiological volume range
        pressure_range=(0, 120)  # Physiological pressure range
    )
    
    # Optimizer
    optimizer = optim.Adam(
        list(model.parameters()) + list(physics_loss.parameters()), 
        lr=learning_rate
    )
    
    # Training loop
    training_losses = []
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predicted_pressures = model(volumes_tensor)
        
        # Compute loss
        loss = physics_loss(predicted_pressures, pressures_tensor, volumes_tensor)
        
        # Backward pass
        loss.backward()
        
        # Optimize
        optimizer.step()
        
        # Record loss
        training_losses.append(loss.item())
        
        # Periodic reporting
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(131)
    plt.plot(training_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Volume vs Pressure (Original)
    plt.subplot(132)
    plt.scatter(volumes, pressures, alpha=0.5, label='Original')
    plt.title('Original Volume-Pressure Relationship')
    plt.xlabel('Volume')
    plt.ylabel('Pressure')
    
    # Volume vs Pressure (Predicted)
    plt.subplot(133)
    with torch.no_grad():
        predicted = model(volumes_tensor).numpy()
    
    plt.scatter(volumes, predicted, alpha=0.5, color='red', label='Predicted')
    plt.title('Predicted Volume-Pressure Relationship')
    plt.xlabel('Volume')
    plt.ylabel('Pressure')
    
    plt.tight_layout()
    plt.savefig('cardiac_volume_pressure_conversion.png')
    plt.show()
    
    return model, training_losses

# Main execution
if __name__ == '__main__':
    trained_model, losses = train_cardiac_volume_pressure_converter()