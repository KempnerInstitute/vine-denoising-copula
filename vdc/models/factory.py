"""Model construction shared by training and pretrained inference."""
from typing import Dict

import torch

from .copula_cnn import CopulaDensityCNN
from .copula_cnn_enhanced import EnhancedCopulaDensityCNN
from .copula_denoiser import CopulaDenoiser
from .unet_grid import GridUNet


def build_model(model_type: str, config: Dict, device: torch.device):
    m = config['data']['m']
    mcfg = config.get('model', {})

    if model_type == 'baseline_cnn':
        return CopulaDensityCNN(
            m=m,
            base_channels=mcfg.get('base_channels', 128),
            n_blocks=mcfg.get('n_blocks', 3),
            dropout=mcfg.get('dropout', 0.1)
        ).to(device)

    if model_type == 'enhanced_cnn':
        use_coords = bool(mcfg.get('use_coordinates', True))
        model = EnhancedCopulaDensityCNN(
            m=m,
            base_channels=mcfg.get('base_channels', 128),
            n_blocks=mcfg.get('n_blocks', 3),
            dropout=mcfg.get('dropout', 0.1),
            input_channels=1 + (2 if use_coords else 0),
            output_mode=mcfg.get('output_mode', 'log'),
            time_conditioning=mcfg.get('time_conditioning', False),
            time_emb_dim=mcfg.get('time_emb_dim', 256),
            multi_scale_aux=mcfg.get('multi_scale_aux', False),
            aux_scales=tuple(mcfg.get('aux_scales', [2, 4]))
        ).to(device)
        # Metadata for downstream inference
        model.vdc_use_coordinates = use_coords
        model.vdc_use_probit_coords = bool(mcfg.get('use_probit_coords', False))
        model.vdc_use_log_n = bool(mcfg.get('use_log_n', False))
        model.vdc_probit_coord_eps = float(mcfg.get('probit_coord_eps', 1e-4))
        model.vdc_transform_to_probit_space = bool(mcfg.get('transform_to_probit_space', False))
        return model

    if model_type == 'denoiser':
        # Compute input_channels: 1 (base) + coordinates (2) + log_n (1)
        use_coords = bool(mcfg.get('use_coordinates', True))
        use_log_n = bool(mcfg.get('use_log_n', False))
        input_ch = 1 + (1 if use_log_n else 0) + (2 if use_coords else 0)
        model = CopulaDenoiser(
            m=m,
            input_channels=input_ch,
            base_channels=mcfg.get('base_channels', 128),
            depth=mcfg.get('depth', mcfg.get('n_blocks', 4)),
            blocks_per_level=mcfg.get('blocks_per_level', 2),
            dropout=mcfg.get('dropout', 0.1),
            time_emb_dim=mcfg.get('time_emb_dim', 256),
            output_mode=mcfg.get('output_mode', 'log'),
        ).to(device)
        # Metadata for downstream inference
        model.vdc_use_coordinates = use_coords
        model.vdc_use_probit_coords = bool(mcfg.get('use_probit_coords', False))
        model.vdc_use_log_n = use_log_n
        model.vdc_probit_coord_eps = float(mcfg.get('probit_coord_eps', 1e-4))
        model.vdc_transform_to_probit_space = bool(mcfg.get('transform_to_probit_space', False))
        return model

    if model_type == 'diffusion_unet':
        # Diffusion UNet supports optional conditioning channels (e.g., log-histogram),
        # controlled via config['model']['in_channels'].
        # Default remains 1 for backward compatibility.
        model = GridUNet(
            m=m,
            in_channels=mcfg.get('in_channels', 1),
            base_channels=mcfg.get('base_channels', 128),
            channel_mults=tuple(mcfg.get('channel_mults', [1, 2, 4])),
            num_res_blocks=mcfg.get('num_res_blocks', 2),
            attention_resolutions=tuple(mcfg.get('attention_resolutions', [])),
            dropout=mcfg.get('dropout', 0.1),
            time_emb_dim=mcfg.get('time_emb_dim', 256)
        ).to(device)
        # Metadata for downstream inference
        model.vdc_use_coordinates = bool(mcfg.get('use_coordinates', False))
        model.vdc_use_probit_coords = bool(mcfg.get('use_probit_coords', False))
        model.vdc_use_log_n = bool(mcfg.get('use_log_n', False))
        model.vdc_probit_coord_eps = float(mcfg.get('probit_coord_eps', 1e-4))
        model.vdc_transform_to_probit_space = bool(mcfg.get('transform_to_probit_space', False))
        return model

    raise ValueError(f"Unknown model type: {model_type}")

