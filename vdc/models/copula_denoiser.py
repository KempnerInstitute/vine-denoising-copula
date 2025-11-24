"""Noise-conditioned single-pass copula denoiser.

This model predicts either:
  (a) log-density directly (output_mode='log')
  (b) noise/residual added to a noisy log-density observation (output_mode='residual')

Differences from diffusion UNet:
  - Accepts optional coordinate channels (raw or probit) like enhanced CNN.
  - Lightweight relative to full UNet; aims for single forward pass at chosen noise level t.
  - Designed to be benchmarked against direct CNN and full diffusion.

Forward Inputs:
  x : (B, C_in, m, m)
  t : (B,) noise level in [0,1]
Returns dict with keys:
  'log_density' if output_mode == 'log'
  'residual'    if output_mode == 'residual'
  always 'features'
"""
from typing import Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return self.mlp(emb)


class FiLMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_gamma = nn.Linear(time_dim, out_ch)
        self.time_beta = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        gamma = self.time_gamma(temb).view(-1, h.shape[1], 1, 1)
        beta = self.time_beta(temb).view(-1, h.shape[1], 1, 1)
        h = F.silu(h * (1 + gamma) + beta)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(F.silu(h))
        return h + self.shortcut(x)


class CopulaDenoiser(nn.Module):
    def __init__(
        self,
        m: int = 256,
        input_channels: int = 1,
        base_channels: int = 128,
        depth: int = 4,
        blocks_per_level: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        output_mode: str = 'log',  # 'log' or 'residual'
    ):
        super().__init__()
        self.m = m
        self.output_mode = output_mode
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU()
        )

        self.in_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)

        # Encoder
        self.enc_levels = nn.ModuleList()
        ch = base_channels
        for lvl in range(depth):
            blocks = []
            for b in range(blocks_per_level):
                blocks.append(FiLMBlock(ch, ch, time_emb_dim, dropout))
            self.enc_levels.append(nn.ModuleList(blocks))
            if lvl < depth - 1:
                self.enc_levels.append(nn.ModuleList([nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]))
                ch *= 2

        # Bottleneck
        self.bottle = FiLMBlock(ch, ch, time_emb_dim, dropout)

        # Decoder
        self.dec_levels = nn.ModuleList()
        for lvl in reversed(range(depth)):
            if lvl < depth - 1:
                self.dec_levels.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
                ch //= 2
            blocks = []
            for b in range(blocks_per_level):
                blocks.append(FiLMBlock(ch, ch, time_emb_dim, dropout))
            self.dec_levels.append(nn.ModuleList(blocks))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        if t.dim() > 1:
            t = t.view(-1)
        temb = self.time_proj(self.time_emb(t))

        h = self.in_conv(x)
        skips = []
        enc_iter = iter(self.enc_levels)
        while True:
            try:
                level_blocks = next(enc_iter)
            except StopIteration:
                break
            # Distinguish between conv-down node (single module) and block list
            if len(level_blocks) == 1 and isinstance(level_blocks[0], nn.Conv2d) and level_blocks[0].stride == (2,2):
                # Downsample
                h = level_blocks[0](h)
                continue
            # Residual blocks
            for blk in level_blocks:
                h = blk(h, temb)
            skips.append(h)

        h = self.bottle(h, temb)

        # Decoder
        dec_iter = iter(self.dec_levels)
        while True:
            try:
                layer = next(dec_iter)
            except StopIteration:
                break
            if isinstance(layer, nn.ConvTranspose2d):
                h = layer(h)
                # concat skip
                if skips:
                    h = h + skips.pop()  # additive skip to save params
                continue
            # Residual blocks list
            for blk in layer:
                h = blk(h, temb)

        raw = self.out_conv(h)
        if self.output_mode == 'log':
            return {'log_density': raw.clamp(min=-20, max=20), 'features': h}
        else:  # residual noise prediction
            return {'residual': raw, 'features': h}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = CopulaDenoiser(input_channels=3, output_mode='log')
    x = torch.randn(2, 3, 128, 128)
    t = torch.rand(2)
    out = model(x, t)
    print(out.keys(), model.count_parameters())