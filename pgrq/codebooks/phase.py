"""
Stage 6: Phase correction (EXPERIMENTAL - default OFF).

Pairs the gd real coordinates into gd/2 complex numbers.
Applies one discrete phase rotation per group to better match the target.

Phase costs log2(R) bits/group where R is the number of discrete phases.
"""

import torch
from typing import Tuple
import math


def to_complex(v: torch.Tensor) -> torch.Tensor:
    """
    Convert real vector to complex by pairing adjacent elements.
    
    v[0::2] + i*v[1::2]
    
    Args:
        v: Real tensor of shape (..., D) where D is even.
        
    Returns:
        z: Complex tensor of shape (..., D/2).
    """
    *batch_dims, D = v.shape
    if D % 2 != 0:
        raise ValueError(f"Dimension must be even, got {D}")
    
    real = v[..., 0::2]  # Even indices
    imag = v[..., 1::2]  # Odd indices
    
    return torch.complex(real, imag)


def to_real(z: torch.Tensor) -> torch.Tensor:
    """
    Convert complex vector back to real by interleaving real and imag.
    
    Args:
        z: Complex tensor of shape (..., D/2).
        
    Returns:
        v: Real tensor of shape (..., D).
    """
    *batch_dims, half_D = z.shape
    D = half_D * 2
    
    v = torch.zeros(*batch_dims, D, device=z.device, dtype=z.real.dtype)
    v[..., 0::2] = z.real
    v[..., 1::2] = z.imag
    
    return v


def compute_phase_candidates(R: int, device: torch.device) -> torch.Tensor:
    """
    Compute R discrete phase rotation factors.
    
    deltas = 2*pi*arange(R)/R
    phases = exp(i*deltas)
    
    Args:
        R: Number of discrete phases (typically 4, 8, or 16).
        device: Target device.
        
    Returns:
        phases: Complex tensor of shape (R,) with unit magnitude.
    """
    deltas = 2 * math.pi * torch.arange(R, device=device, dtype=torch.float32) / R
    phases = torch.complex(torch.cos(deltas), torch.sin(deltas))
    return phases


def find_best_phase(
    z_target: torch.Tensor,
    z_centroid: torch.Tensor,
    R: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the best discrete phase to rotate centroid toward target.
    
    For each row, find r* = argmin_r || z_target - exp(i*2*pi*r/R) * z_centroid ||^2
    
    Args:
        z_target: Complex target of shape (N, D/2).
        z_centroid: Complex centroid of shape (N, D/2).
        R: Number of discrete phases.
        
    Returns:
        best_r: Best phase indices of shape (N,).
        rotated: Rotated centroids of shape (N, D/2).
    """
    N = z_target.shape[0]
    device = z_target.device
    
    # Get phase candidates
    phases = compute_phase_candidates(R, device)  # (R,)
    
    # Compute all rotated candidates: (R, N, D/2)
    # phases: (R,) -> (R, 1, 1)
    # z_centroid: (N, D/2) -> (1, N, D/2)
    candidates = phases.view(R, 1, 1) * z_centroid.unsqueeze(0)  # (R, N, D/2)
    
    # Compute squared errors
    # z_target: (N, D/2) -> (1, N, D/2)
    errors = (candidates - z_target.unsqueeze(0)).abs() ** 2  # (R, N, D/2)
    errors = errors.sum(dim=-1)  # (R, N)
    
    # Find best phase for each row
    best_r = errors.argmin(dim=0)  # (N,)
    
    # Get the rotated centroids for the best phases
    best_phases = phases[best_r]  # (N,)
    rotated = best_phases.unsqueeze(-1) * z_centroid  # (N, D/2)
    
    return best_r, rotated


class PhaseCorrector:
    """
    Phase correction module (Stage 6).
    
    Applies discrete phase rotations to better match centroids to targets.
    This is EXPERIMENTAL and OFF by default because it may not beat
    spending the same bits on residual codebooks.
    """
    
    def __init__(self, R: int = 8, device: str = "cpu"):
        """
        Initialize phase corrector.
        
        Args:
            R: Number of discrete phases (4, 8, or 16).
            device: Target device.
        """
        if R not in [4, 8, 16]:
            raise ValueError(f"R must be 4, 8, or 16, got {R}")
        
        self.R = R
        self.device = device
        self.bits_per_group = math.log2(R)
    
    def apply_phase(
        self,
        target: torch.Tensor,
        centroid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply phase correction to centroids.
        
        Args:
            target: Target vectors of shape (N, D) where D is even.
            centroid: Base centroid vectors of shape (N, D).
            
        Returns:
            phase_codes: Phase indices of shape (N,) in [0, R).
            corrected: Phase-corrected centroids of shape (N, D).
        """
        # Convert to complex
        z_target = to_complex(target)
        z_centroid = to_complex(centroid)
        
        # Find best phase
        phase_codes, z_corrected = find_best_phase(z_target, z_centroid, self.R)
        
        # Convert back to real
        corrected = to_real(z_corrected)
        
        return phase_codes, corrected
    
    def decode_phase(
        self,
        centroid: torch.Tensor,
        phase_codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply stored phase codes to centroids during decoding.
        
        Args:
            centroid: Base centroid vectors of shape (N, D).
            phase_codes: Phase indices of shape (N,) in [0, R).
            
        Returns:
            corrected: Phase-corrected centroids of shape (N, D).
        """
        # Get phase factors
        phases = compute_phase_candidates(self.R, centroid.device)
        selected_phases = phases[phase_codes]  # (N,)
        
        # Convert centroid to complex, apply phase, convert back
        z_centroid = to_complex(centroid)
        z_corrected = selected_phases.unsqueeze(-1) * z_centroid
        corrected = to_real(z_corrected)
        
        return corrected
    
    def get_improvement(
        self,
        target: torch.Tensor,
        centroid: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute MSE improvement from phase correction.
        
        Args:
            target: Target vectors of shape (N, D).
            centroid: Base centroid vectors of shape (N, D).
            
        Returns:
            mse_before: MSE without phase correction.
            mse_after: MSE with phase correction.
        """
        mse_before = ((target - centroid) ** 2).mean().item()
        
        _, corrected = self.apply_phase(target, centroid)
        mse_after = ((target - corrected) ** 2).mean().item()
        
        return mse_before, mse_after
