"""
Simple CNN model as a workaround for buggy UNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for density estimation - replacement for buggy UNet.
    
    Takes as input:
    - Density grid: (B, 1, m, m)
    - Time embedding: (B,)
    
    Outputs:
    - Log-density grid: (B, 1, m, m)
    """
    
    def __init__(
        self,
        m: int = 64,
        in_channels: int = 1,
        base_channels: int = 128,
        num_blocks: int = 4,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.m = m
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Main blocks
        self.blocks = nn.ModuleList([
            ResBlock(base_channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        
        # Output
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input grid (B, 1, m, m)
            t: Time (B,), values in [0, 1]
            
        Returns:
            Output grid (B, 1, m, m)
        """
        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        t_emb = self.time_embed(t)  # (B, time_emb_dim)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Main blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, channels)
        
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t_proj = self.time_proj(t_emb)[:, :, None, None]  # (B, C, 1, 1)
        h = h + t_proj
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return x + h  # Residual connection


if __name__ == "__main__":
    # Test the model
    model = SimpleCNN(m=64, base_channels=128, num_blocks=4)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 64, 64)
    t = torch.rand(2)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"✓ Success!")
