"""
U-Net architecture for grid-based copula density estimation.

This module implements a 2D U-Net that takes a histogram and time embedding
and outputs a log-density grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion models.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (B,) or (B, 1, 1, 1) with values in [0, 1]
            
        Returns:
            Tensor of shape (B, dim) with sinusoidal embeddings
        """
        t = t.squeeze()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class ResBlock(nn.Module):
    """
    Residual block with time embedding conditioning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
            t_emb: (B, time_emb_dim)
            
        Returns:
            (B, out_channels, H, W)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding (broadcast over spatial dimensions)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for U-Net.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = attn @ v
        
        # Reshape back
        h = h.transpose(2, 3).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        downsample: bool = True,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
            )
            for i in range(num_res_blocks)
        ])
        
        if use_attention:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = None
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        for block in self.res_blocks:
            x = block(x, t_emb)
        
        if self.attn is not None:
            x = self.attn(x)
        
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x_down, x  # Return both downsampled and skip connection
        else:
            return x, x


class UpBlock(nn.Module):
    """Upsampling block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        upsample: bool = True,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
            )
            for i in range(num_res_blocks)
        ])
        
        if use_attention:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = None
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.upsample = None
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        # Upsample BEFORE concatenating with skip
        if self.upsample is not None:
            x = self.upsample(x)
        
        # Now concatenate skip connection (both should have same spatial size)
        x = torch.cat([x, skip], dim=1)
        
        for block in self.res_blocks:
            x = block(x, t_emb)
        
        if self.attn is not None:
            x = self.attn(x)
        
        return x


class GridUNet(nn.Module):
    """
    U-Net for transforming histogram to log-density grid.
    
    Takes as input:
    - Histogram: (B, 1, m, m)
    - Time embedding: (B,) or (B, 1, 1, 1)
    
    Outputs:
    - Log-density grid: (B, 1, m, m)
    """
    
    def __init__(
        self,
        m: int = 64,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.m = m
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            for _ in range(num_res_blocks):
                use_attn = (m // (2 ** i)) in attention_resolutions
                self.down_blocks.append(
                    DownBlock(
                        now_channels,
                        out_channels,
                        time_emb_dim,
                        num_res_blocks=1,
                        use_attention=use_attn,
                        downsample=False,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            if i < len(channel_mults) - 1:
                self.down_blocks.append(
                    DownBlock(
                        now_channels,
                        now_channels,
                        time_emb_dim,
                        num_res_blocks=1,
                        use_attention=False,
                        downsample=True,
                    )
                )
                channels.append(now_channels)
        
        # Middle
        self.mid_block1 = ResBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResBlock(now_channels, now_channels, time_emb_dim)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                use_attn = (m // (2 ** (len(channel_mults) - i - 1))) in attention_resolutions
                
                self.up_blocks.append(
                    UpBlock(
                        now_channels + skip_channels,
                        out_channels,
                        time_emb_dim,
                        num_res_blocks=1,
                        use_attention=use_attn,
                        upsample=(j == num_res_blocks and i < len(channel_mults) - 1),
                    )
                )
                now_channels = out_channels
        
        # Output
        self.norm_out = nn.GroupNorm(8, now_channels)
        self.conv_out = nn.Conv2d(now_channels, 1, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Histogram (B, 1, m, m)
            t: Time (B,) or (B, 1, 1, 1), values in [0, 1]
            
        Returns:
            Log-density grid (B, 1, m, m)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Downsampling with skip connections
        skips = [h]  # Include initial features
        for block in self.down_blocks:
            h, skip = block(h, t_emb)
            skips.append(skip)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Upsampling - reverse skips and match properly
        skips = skips[::-1]  # Reverse to match upsampling order
        for i, block in enumerate(self.up_blocks):
            skip = skips[i] if i < len(skips) else h
            h = block(h, skip, t_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


if __name__ == "__main__":
    # Test the model
    model = GridUNet(m=64, base_channels=64)
    
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
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
