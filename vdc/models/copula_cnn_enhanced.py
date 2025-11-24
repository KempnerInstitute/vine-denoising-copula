"""Enhanced Copula CNN with optional log-density head, coordinate/probit inputs, and time/noise conditioning.

Design goals:
- Backward-compatible with baseline model usage via dict outputs.
- Optional coordinate channels (u,v) or probit transformed coordinates Φ^{-1}(u), Φ^{-1}(v) to ease boundary learning.
- Output modes: 'softplus' (positive density) or 'log' (raw log-density) for better numerical stability.
- Optional time conditioning (scalar t \in [0,1]) to allow unified denoiser / noise-level aware training.

Forward contract:
Inputs:
    x: (B, C_in, m, m) where C_in = 1 (noise baseline) + 2 (coords) if enabled.
    t (optional): (B,) or (B,1) in [0,1] if time_conditioning=True.
Outputs:
    dict with keys:
        'density' if output_mode == 'softplus'
        'log_density' if output_mode == 'log'
        always includes 'features' for potential auxiliary losses.
"""
from typing import Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal + MLP time embedding similar to diffusion UNet architectures."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Ensure shape (B,)
        if t.dim() > 1:
            t = t.view(-1)
        half = self.dim // 2
        # log space frequencies
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # pad in case of odd
            emb = F.pad(emb, (0,1), value=0.0)
        return self.mlp(emb)  # (B, dim)


class FiLMResidualBlock(nn.Module):
    """Residual block with FiLM modulation from time embedding (optional)."""
    def __init__(self, channels: int, dropout: float = 0.1, time_dim: Optional[int] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.use_time = time_dim is not None
        if self.use_time:
            self.film_gamma = nn.Linear(time_dim, channels)
            self.film_beta = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        if self.use_time and temb is not None:
            gamma = self.film_gamma(temb).view(-1, h.shape[1], 1, 1)
            beta = self.film_beta(temb).view(-1, h.shape[1], 1, 1)
            h = h * (1 + gamma) + beta
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.bn2(h)
        out = F.relu(h + residual)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(attention))
        return x * attention


class ChannelAttention(nn.Module):
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


class EnhancedCopulaDensityCNN(nn.Module):
    def __init__(
        self,
        m: int = 256,
        base_channels: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
        input_channels: int = 1,
        output_mode: str = 'softplus',  # 'softplus' or 'log'
        time_conditioning: bool = False,
        time_emb_dim: int = 256,
        multi_scale_aux: bool = False,
        aux_scales: tuple = (2,4),
    ):
        super().__init__()
        self.m = m
        self.output_mode = output_mode
        self.time_conditioning = time_conditioning
        self.time_emb_dim = time_emb_dim if time_conditioning else None

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.SiLU()
        )

        # Encoder blocks
        self.encoder1 = self._make_block(base_channels, n_blocks)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.encoder2 = self._make_block(base_channels * 2, n_blocks)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.encoder3 = self._make_block(base_channels * 4, n_blocks)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            FiLMResidualBlock(base_channels * 8, dropout, self.time_emb_dim) for _ in range(n_blocks)
        ])
        self.bottle_attn = nn.Sequential(
            ChannelAttention(base_channels * 8),
            SpatialAttention(base_channels * 8)
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3_conv = nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1)
        self.decoder3 = self._make_block(base_channels * 4, n_blocks)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2_conv = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.decoder2 = self._make_block(base_channels * 2, n_blocks)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1_conv = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.decoder1 = self._make_block(base_channels, n_blocks)

        # Output heads (main resolution)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.SiLU(),
        )
        self.density_head = nn.Conv2d(base_channels // 2, 1, 1)
        if output_mode == 'softplus':
            self.output_activation = nn.Softplus()
        elif output_mode == 'log':
            self.output_activation = nn.Identity()  # produce raw log-density
        else:
            raise ValueError(f"Unknown output_mode={output_mode}")

        # Multi-scale auxiliary heads (predict coarse log-density / density directly for deep supervision)
        self.multi_scale_aux = multi_scale_aux
        self.aux_scales = aux_scales if multi_scale_aux else ()
        if self.multi_scale_aux:
            self.aux_heads = nn.ModuleDict()
            for s in self.aux_scales:
                self.aux_heads[f's{s}'] = nn.Sequential(
                    nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(base_channels // 4, 1, 1)
                )

        # Time embedding if used
        if self.time_conditioning:
            self.time_mlp = TimeEmbedding(self.time_emb_dim)

        self._initialize_weights()

    def _make_block(self, channels: int, n_blocks: int):
        return nn.Sequential(*[FiLMResidualBlock(channels, dropout=0.1, time_dim=self.time_emb_dim) for _ in range(n_blocks)])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.time_conditioning:
            if t is None:
                # default zero time (could also randomize outside)
                t = torch.zeros(x.shape[0], device=x.device)
            temb = self.time_mlp(t)  # (B, time_emb_dim)
        else:
            temb = None

        # Encoder
        x1 = self.input_conv(x)
        x2 = self.encoder1(x1 if temb is None else self._apply_time(x1, temb))
        x3 = F.relu(self.down1(x2))
        x4 = self.encoder2(x3 if temb is None else self._apply_time(x3, temb))
        x5 = F.relu(self.down2(x4))
        x6 = self.encoder3(x5 if temb is None else self._apply_time(x5, temb))
        x7 = F.relu(self.down3(x6))

        h = x7
        for blk in self.bottleneck:
            h = blk(h, temb)
        h = self.bottle_attn(h)

        # Decoder with skip connections
        d3 = self.up3(h)
        d3 = torch.cat([d3, x6], dim=1)
        d3 = F.relu(self.dec3_conv(d3))
        d3 = self.decoder3(d3 if temb is None else self._apply_time(d3, temb))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x4], dim=1)
        d2 = F.relu(self.dec2_conv(d2))
        d2 = self.decoder2(d2 if temb is None else self._apply_time(d2, temb))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x2], dim=1)
        d1 = F.relu(self.dec1_conv(d1))
        d1 = self.decoder1(d1 if temb is None else self._apply_time(d1, temb))

        feats = self.out_conv(d1)
        raw_out = self.density_head(feats)
        out: Dict[str, torch.Tensor] = {'features': feats}
        if self.output_mode == 'softplus':
            out['density'] = self.output_activation(raw_out)
        else:
            # Clamp to reasonable range to prevent overflow in exp()
            # Match training script clamping: [-15, 15] => exp range [3e-7, 3.3e6]
            # This prevents numerical overflow when converting log → density
            out['log_density'] = raw_out.clamp(min=-15, max=15)

        # Auxiliary multi-scale predictions (downsample feats then predict)
        if self.multi_scale_aux:
            for s in self.aux_scales:
                if feats.shape[-1] % s != 0:
                    continue
                ds = torch.nn.functional.avg_pool2d(feats, s)
                head_key = f's{s}'
                if head_key not in self.aux_heads:
                    continue
                head = self.aux_heads[head_key]
                aux_raw = head(ds)
                if self.output_mode == 'softplus':
                    out[f'aux_s{s}'] = self.output_activation(aux_raw)
                else:
                    out[f'aux_log_s{s}'] = aux_raw.clamp(min=-20, max=20)
        return out

    def _apply_time(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # Simple additive conditioning broadcast
        return x + temb.view(temb.shape[0], -1, 1, 1)[:, : x.shape[1]]  # crop if mismatch

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = EnhancedCopulaDensityCNN(m=128, input_channels=3, output_mode='log', time_conditioning=True)
    x = torch.randn(2, 3, 128, 128)
    t = torch.rand(2)
    out = model(x, t)
    print('keys:', out.keys())
    if 'log_density' in out:
        print('log range', out['log_density'].min().item(), out['log_density'].max().item())
    print('params', model.count_parameters())
