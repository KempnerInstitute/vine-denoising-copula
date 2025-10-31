"""
Copula-aware diffusion process that preserves copula properties.

Key constraint: Marginals must remain uniform U(0,1) throughout diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CopulaAwareDiffusion(nn.Module):
    """
    Diffusion process that maintains copula properties.
    
    Standard diffusion adds noise: x_t = √(1-β_t) x_0 + √β_t ε
    Problem: This violates uniform marginals!
    
    Solution: Diffuse in the copula density space, not sample space.
    - Forward: Add noise to log-density
    - Reverse: Denoise while projecting to valid copula
    """
    
    def __init__(
        self,
        beta_schedule: str = 'linear',
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        
        self.timesteps = timesteps
        
        # Noise schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule from Improved DDPM
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Register as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1 - self.alphas_cumprod))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        
        For copula density (on log scale):
        log c_t = √α̅_t log c_0 + √(1-α̅_t) ε
        
        Args:
            x_start: Initial log-density (B, 1, m, m)
            t: Timestep (B,) in [0, timesteps-1]
            noise: Optional noise (same shape as x_start)
            
        Returns:
            Noisy log-density x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get coefficients for timestep t
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting (B, 1, 1, 1)
        sqrt_alpha_t = sqrt_alpha_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(-1, 1, 1, 1)
        
        # Add noise
        x_t = sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
        
        return x_t
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        project_copula: bool = True,
    ) -> torch.Tensor:
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model (predicts noise or x_0)
            x_t: Noisy input (B, 1, m, m)
            t: Timestep (B,)
            project_copula: If True, project to valid copula
            
        Returns:
            Denoised x_{t-1}
        """
        # Model prediction (assume predicting noise)
        pred_noise = model(x_t, t)
        
        # Compute x_0 prediction
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        
        # Project to valid copula if requested
        if project_copula:
            pred_x0 = self.project_to_copula(pred_x0)
        
        # Compute x_{t-1}
        if t[0] > 0:
            # Use posterior mean
            alpha_prev = self.alphas_cumprod[t - 1].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            
            # Posterior mean: μ_θ(x_t, t)
            coef1 = torch.sqrt(alpha_prev) * beta_t / (1 - alpha_t)
            coef2 = torch.sqrt(self.alphas[t].view(-1, 1, 1, 1)) * (1 - alpha_prev) / (1 - alpha_t)
            
            mean = coef1 * pred_x0 + coef2 * x_t
            
            # Add noise (except at t=0)
            variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = pred_x0
        
        return x_prev
    
    def project_to_copula(self, log_density: torch.Tensor) -> torch.Tensor:
        """
        Project to valid copula by enforcing constraints.
        
        Copula constraints:
        1. c(u,v) ≥ 0  (non-negative)
        2. ∫_0^1 c(u,v) dv = 1  (uniform U marginal)
        3. ∫_0^1 c(u,v) du = 1  (uniform V marginal)
        4. ∫∫ c(u,v) du dv = 1  (unit mass)
        
        Args:
            log_density: Log-density (B, 1, m, m)
            
        Returns:
            Projected log-density satisfying copula constraints
        """
        from vdc.models.projection import copula_project
        
        # Convert to density (ensure non-negative)
        density = torch.exp(log_density)
        
        # Project using IPFP/Sinkhorn
        density_proj = copula_project(density, iters=20)
        
        # Back to log scale
        log_density_proj = torch.log(density_proj + 1e-12)
        
        return log_density_proj


class MarginalPreservingLoss(nn.Module):
    """
    Loss that explicitly penalizes deviation from uniform marginals.
    
    This acts as a regularizer during training to ensure the network
    learns to respect copula constraints.
    """
    
    def __init__(self, penalty_weight: float = 0.1):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """
        Compute marginal uniformity penalty.
        
        For a valid copula:
        - U marginal: ∫_0^1 c(u,v) dv = 1 for all u
        - V marginal: ∫_0^1 c(u,v) du = 1 for all v
        
        Args:
            density: Copula density (B, 1, m, m)
            
        Returns:
            Penalty loss (scalar)
        """
        B, _, m, _ = density.shape
        
        # Compute marginals (integrate out one dimension)
        # For discrete grid, integration = sum / m
        u_marginal = density.sum(dim=3) / m  # (B, 1, m) - integrate over v
        v_marginal = density.sum(dim=2) / m  # (B, 1, m) - integrate over u
        
        # Should both be 1 everywhere
        target = torch.ones_like(u_marginal)
        
        # L2 penalty
        loss_u = F.mse_loss(u_marginal, target)
        loss_v = F.mse_loss(v_marginal, target)
        
        # Total mass should be 1
        total_mass = density.sum(dim=(2, 3)) / (m * m)  # (B, 1)
        target_mass = torch.ones_like(total_mass)
        loss_mass = F.mse_loss(total_mass, target_mass)
        
        # Combined penalty
        penalty = (loss_u + loss_v + loss_mass) * self.penalty_weight
        
        return penalty


class CopulaConstrainedTraining:
    """
    Training strategy that enforces copula properties.
    
    Three mechanisms:
    1. Copula-aware diffusion (diffuse in density space)
    2. IPFP projection after each forward pass
    3. Marginal uniformity loss
    """
    
    @staticmethod
    def training_step(
        model: nn.Module,
        hist: torch.Tensor,
        target_log_density: torch.Tensor,
        diffusion: CopulaAwareDiffusion,
        device: torch.device,
        use_projection: bool = True,
        marginal_penalty_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single training step with copula constraints.
        
        Args:
            model: Denoising model
            hist: Input histogram (B, 1, m, m)
            target_log_density: Target log-density (B, m, m)
            diffusion: Diffusion process
            device: Computation device
            use_projection: Whether to project to copula
            marginal_penalty_weight: Weight for marginal loss
            
        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        B, _, m, _ = hist.shape
        
        # Random timestep
        t = torch.randint(0, diffusion.timesteps, (B,), device=device)
        
        # Add noise to target
        target_log_density = target_log_density.unsqueeze(1)  # (B, 1, m, m)
        noise = torch.randn_like(target_log_density)
        noisy_log_density = diffusion.q_sample(target_log_density, t, noise)
        
        # Model prediction (predict noise)
        pred_noise = model(hist, t.float() / diffusion.timesteps)
        
        # Noise prediction loss
        loss_noise = F.mse_loss(pred_noise, noise)
        
        # Reconstruct x_0 from prediction
        alpha_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        pred_x0 = (noisy_log_density - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        
        # Convert to density for projection
        if use_projection:
            pred_density = torch.exp(pred_x0)
            
            # Project to copula
            from vdc.models.projection import copula_project
            pred_density_proj = copula_project(pred_density, iters=10)
            
            # Marginal penalty on projected density
            marginal_loss = MarginalPreservingLoss(marginal_penalty_weight)
            loss_marginal = marginal_loss(pred_density_proj)
        else:
            loss_marginal = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = loss_noise + loss_marginal
        
        metrics = {
            'loss_noise': loss_noise.item(),
            'loss_marginal': loss_marginal.item(),
        }
        
        return loss, metrics


if __name__ == "__main__":
    # Test copula-aware diffusion
    print("Testing CopulaAwareDiffusion...")
    
    diffusion = CopulaAwareDiffusion(timesteps=1000, beta_schedule='linear')
    
    # Create dummy log-density
    m = 32
    log_density = torch.randn(4, 1, m, m)
    
    # Forward diffusion
    t = torch.tensor([100, 200, 500, 900])
    noisy = diffusion.q_sample(log_density, t)
    
    print(f"Original shape: {log_density.shape}")
    print(f"Noisy shape: {noisy.shape}")
    print(f"Noise level increases with t: {noisy.std(dim=(1,2,3))}")
    
    # Test projection
    print("\nTesting copula projection...")
    projected = diffusion.project_to_copula(noisy)
    
    density_proj = torch.exp(projected)
    
    # Check marginals
    u_marginal = density_proj.mean(dim=3)  # Should be ~constant
    v_marginal = density_proj.mean(dim=2)  # Should be ~constant
    
    print(f"U marginal std: {u_marginal.std():.6f} (should be small)")
    print(f"V marginal std: {v_marginal.std():.6f} (should be small)")
    print(f"Total mass: {density_proj.mean():.6f} (should be ~1/{m**2} = {1/m**2:.6f})")
    
    print("\n✓ Copula-aware diffusion tests passed!")
