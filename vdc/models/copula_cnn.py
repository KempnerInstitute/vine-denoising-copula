"""
CNN for direct copula density estimation.

This module provides a deeper residual architecture with attention and skip
connections for experiments that do not use the main released U-Net path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x + residual  # Skip connection
        x = F.relu(x)
        return x


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important regions."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        
    def forward(self, x):
        # Channel-wise attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(attention))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CopulaDensityCNN(nn.Module):
    """
    CNN for direct copula density estimation.
    
    Architecture:
    - Deep encoder with residual blocks
    - Multi-scale feature extraction
    - Attention mechanisms
    - Skip connections similar to UNet
    - Dense decoder with residual blocks
    
    Args:
        m: Grid resolution (e.g., 256)
        base_channels: Base number of channels (default: 128)
        n_blocks: Number of residual blocks per level (default: 3)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, m=256, base_channels=128, n_blocks=3, dropout=0.1):
        super().__init__()
        self.m = m
        
        # Initial projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # Encoder path with residual blocks
        self.encoder1 = self._make_encoder_block(base_channels, base_channels, n_blocks, dropout)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        
        self.encoder2 = self._make_encoder_block(base_channels * 2, base_channels * 2, n_blocks, dropout)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        
        self.encoder3 = self._make_encoder_block(base_channels * 4, base_channels * 4, n_blocks, dropout)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(base_channels * 8, dropout) for _ in range(n_blocks)],
            ChannelAttention(base_channels * 8),
            SpatialAttention(base_channels * 8)
        )
        
        # Decoder path with skip connections
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = self._make_decoder_block(base_channels * 8, base_channels * 4, n_blocks, dropout)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = self._make_decoder_block(base_channels * 4, base_channels * 2, n_blocks, dropout)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = self._make_decoder_block(base_channels * 2, base_channels, n_blocks, dropout)
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Softplus()  # Ensure positive output for density
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_encoder_block(self, in_channels, out_channels, n_blocks, dropout):
        layers = []
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
        for _ in range(n_blocks):
            layers.append(ResidualBlock(out_channels, dropout))
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_channels, out_channels, n_blocks, dropout):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(out_channels, dropout))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, m, m) - can be zeros, noise, or initial estimate
            
        Returns:
            Predicted copula density (B, 1, m, m)
        """
        # Encoder with skip connections
        x1 = self.input_conv(x)
        x2 = self.encoder1(x1)
        
        x3 = F.relu(self.down1(x2))
        x4 = self.encoder2(x3)
        
        x5 = F.relu(self.down2(x4))
        x6 = self.encoder3(x5)
        
        x7 = F.relu(self.down3(x6))
        
        # Bottleneck
        x_bottle = self.bottleneck(x7)
        
        # Decoder with skip connections
        x8 = self.up3(x_bottle)
        x8 = torch.cat([x8, x6], dim=1)  # Skip connection
        x9 = self.decoder3(x8)
        
        x10 = self.up2(x9)
        x10 = torch.cat([x10, x4], dim=1)  # Skip connection
        x11 = self.decoder2(x10)
        
        x12 = self.up1(x11)
        x12 = torch.cat([x12, x2], dim=1)  # Skip connection
        x13 = self.decoder1(x12)
        
        # Output
        density = self.output_conv(x13)
        
        return density
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    model = CopulaDensityCNN(m=256, base_channels=128, n_blocks=3)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
