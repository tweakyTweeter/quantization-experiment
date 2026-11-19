"""
Stage 8: Mixed-precision allocation using Lagrangian optimization.

Allocates different precision modes to different groups based on:
- Importance (activation-aware): I = E[a^2] * ||w||^2
- Error-bit tradeoff per mode

Uses binary search over Lagrange multiplier to hit target bit budget.
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class PrecisionMode(Enum):
    """Precision modes for mixed-precision allocation."""
    BASE = 0                   # Base codebook only
    BASE_PHASE = 1             # Base + phase correction
    BASE_RESID = 2             # Base + residual codebooks
    BASE_PHASE_RESID = 3       # Base + phase + residual
    BASE_RESID_DIGITS = 4      # Base + residual + scalar floor


@dataclass
class ModeConfig:
    """Configuration for a precision mode."""
    mode: PrecisionMode
    bits: float                # Bits per group for this mode
    
    # These will be filled during allocation
    error: Optional[float] = None  # Expected MSE for this mode


class MixedPrecisionAllocator:
    """
    Lagrangian-based mixed-precision allocator (Stage 8).
    
    For each group, chooses the mode that minimizes:
        I_g * error_g(mode) + lambda * bits_g(mode)
    
    Binary search over lambda to hit target bit budget.
    """
    
    def __init__(
        self,
        base_codebook_size: int = 256,
        num_residual_levels: int = 2,
        residual_codebook_size: int = 256,
        phase_resolution: int = 8,
        scalar_floor_bits: int = 2,
        group_dim: int = 16,
        phase_enabled: bool = False,
        scalar_floor_enabled: bool = False,
    ):
        """
        Initialize allocator with mode configurations.
        
        Args:
            base_codebook_size: K0 for base codebook.
            num_residual_levels: M for residual levels.
            residual_codebook_size: K for residual codebooks.
            phase_resolution: R for phase correction.
            scalar_floor_bits: Bits per coord for scalar floor.
            group_dim: Dimension of each group (gd).
            phase_enabled: Whether phase mode is available.
            scalar_floor_enabled: Whether scalar floor mode is available.
        """
        self.base_bits = math.log2(base_codebook_size)
        self.phase_bits = math.log2(phase_resolution) if phase_enabled else 0
        self.resid_bits = num_residual_levels * math.log2(residual_codebook_size)
        self.scalar_bits = group_dim * scalar_floor_bits
        
        self.phase_enabled = phase_enabled
        self.scalar_floor_enabled = scalar_floor_enabled
        
        # Build available modes
        self.available_modes = self._build_modes()
    
    def _build_modes(self) -> List[ModeConfig]:
        """Build list of available precision modes with their bit costs."""
        modes = []
        
        # BASE: always available
        modes.append(ModeConfig(
            mode=PrecisionMode.BASE,
            bits=self.base_bits,
        ))
        
        # BASE_PHASE: only if phase enabled
        if self.phase_enabled:
            modes.append(ModeConfig(
                mode=PrecisionMode.BASE_PHASE,
                bits=self.base_bits + self.phase_bits,
            ))
        
        # BASE_RESID: always available (main mode)
        modes.append(ModeConfig(
            mode=PrecisionMode.BASE_RESID,
            bits=self.base_bits + self.resid_bits,
        ))
        
        # BASE_PHASE_RESID: only if phase enabled
        if self.phase_enabled:
            modes.append(ModeConfig(
                mode=PrecisionMode.BASE_PHASE_RESID,
                bits=self.base_bits + self.phase_bits + self.resid_bits,
            ))
        
        # BASE_RESID_DIGITS: only if scalar floor enabled
        if self.scalar_floor_enabled:
            modes.append(ModeConfig(
                mode=PrecisionMode.BASE_RESID_DIGITS,
                bits=self.base_bits + self.resid_bits + self.scalar_bits,
            ))
        
        return modes
    
    def compute_mode_errors(
        self,
        groups: torch.Tensor,
        base_codebook: "KMeansCodebook",
        rvq_codebook: Optional["RVQCodebook"] = None,
        phase_corrector: Optional["PhaseCorrector"] = None,
        scalar_quantizer: Optional[object] = None,
    ) -> Dict[PrecisionMode, torch.Tensor]:
        """
        Compute reconstruction error for each mode on each group.
        
        Args:
            groups: Group data of shape (N, gd).
            base_codebook: Trained base codebook.
            rvq_codebook: Trained RVQ codebook (optional).
            phase_corrector: Phase corrector (optional).
            scalar_quantizer: Scalar floor quantizer (optional).
            
        Returns:
            errors: Dict mapping mode -> error tensor of shape (N,).
        """
        N, gd = groups.shape
        device = groups.device
        errors = {}
        
        # BASE: just base codebook reconstruction
        base_codes = base_codebook.encode(groups)
        base_recon = base_codebook.decode(base_codes)
        base_error = ((groups - base_recon) ** 2).sum(dim=-1)  # (N,)
        errors[PrecisionMode.BASE] = base_error
        
        # BASE_PHASE: base + phase correction
        if self.phase_enabled and phase_corrector is not None:
            _, phase_recon = phase_corrector.apply_phase(groups, base_recon)
            phase_error = ((groups - phase_recon) ** 2).sum(dim=-1)
            errors[PrecisionMode.BASE_PHASE] = phase_error
        
        # BASE_RESID: base + RVQ on residual
        if rvq_codebook is not None:
            residual = groups - base_recon
            rvq_codes = rvq_codebook.encode(residual)
            rvq_recon = rvq_codebook.decode(rvq_codes)
            resid_error = ((groups - base_recon - rvq_recon) ** 2).sum(dim=-1)
            errors[PrecisionMode.BASE_RESID] = resid_error
            
            # BASE_PHASE_RESID: phase first, then RVQ on phase-corrected residual
            if self.phase_enabled and phase_corrector is not None:
                _, phase_recon = phase_corrector.apply_phase(groups, base_recon)
                residual_after_phase = groups - phase_recon
                rvq_codes_p = rvq_codebook.encode(residual_after_phase)
                rvq_recon_p = rvq_codebook.decode(rvq_codes_p)
                phase_resid_error = ((groups - phase_recon - rvq_recon_p) ** 2).sum(dim=-1)
                errors[PrecisionMode.BASE_PHASE_RESID] = phase_resid_error
            
            # BASE_RESID_DIGITS: RVQ + scalar floor on final residual
            if self.scalar_floor_enabled and scalar_quantizer is not None:
                final_residual = groups - base_recon - rvq_recon
                _, scalar_recon = scalar_quantizer.quantize(final_residual.flatten())
                scalar_recon = scalar_recon.view(N, gd)
                digits_error = ((groups - base_recon - rvq_recon - scalar_recon) ** 2).sum(dim=-1)
                errors[PrecisionMode.BASE_RESID_DIGITS] = digits_error
        else:
            # If no RVQ, just use base error for resid modes
            errors[PrecisionMode.BASE_RESID] = base_error
        
        return errors
    
    def allocate(
        self,
        importances: torch.Tensor,
        errors: Dict[PrecisionMode, torch.Tensor],
        target_bits_per_group: float,
        num_precision_classes: int = 4,
    ) -> Tuple[torch.Tensor, Dict[int, PrecisionMode]]:
        """
        Allocate precision modes to groups using Lagrangian optimization.
        
        Minimizes: sum_g I_g * error_g(mode_g)
        Subject to: sum_g bits_g(mode_g) / N <= target_bits_per_group
        
        Args:
            importances: Importance weights of shape (N,).
            errors: Dict mapping mode -> error tensor of shape (N,).
            target_bits_per_group: Target average bits per group.
            num_precision_classes: Number of discrete precision classes for packing.
            
        Returns:
            mode_assignments: Mode indices of shape (N,).
            class_to_mode: Mapping from class index to PrecisionMode.
        """
        N = importances.shape[0]
        device = importances.device
        
        # Build mode tensors
        available = [m for m in self.available_modes if m.mode in errors]
        num_modes = len(available)
        
        bits_tensor = torch.tensor([m.bits for m in available], device=device)
        error_tensor = torch.stack([errors[m.mode] for m in available], dim=0)  # (num_modes, N)
        
        # Binary search for lambda
        target_total_bits = target_bits_per_group * N
        
        lambda_low = 0.0
        lambda_high = 1000.0
        
        best_assignments = None
        best_total_bits = 0
        
        for _ in range(50):  # Binary search iterations
            lambda_mid = (lambda_low + lambda_high) / 2
            
            # Compute cost for each mode at each group
            # cost[m, g] = I_g * error[m, g] + lambda * bits[m]
            cost = importances.unsqueeze(0) * error_tensor + lambda_mid * bits_tensor.unsqueeze(1)
            
            # Choose best mode for each group
            assignments = cost.argmin(dim=0)  # (N,)
            
            # Compute total bits
            total_bits = bits_tensor[assignments].sum().item()
            
            if abs(total_bits - target_total_bits) < 0.01 * target_total_bits:
                best_assignments = assignments
                best_total_bits = total_bits
                break
            elif total_bits > target_total_bits:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
            
            best_assignments = assignments
            best_total_bits = total_bits
        
        # Quantize to precision classes for SIMD-regular packing
        # Group modes into classes based on bit cost
        mode_to_class = {}
        sorted_modes = sorted(enumerate(available), key=lambda x: x[1].bits)
        
        if num_precision_classes >= num_modes:
            # One class per mode
            for i, (mode_idx, mode_cfg) in enumerate(sorted_modes):
                mode_to_class[mode_idx] = i
        else:
            # Bin modes into classes
            modes_per_class = (num_modes + num_precision_classes - 1) // num_precision_classes
            for i, (mode_idx, mode_cfg) in enumerate(sorted_modes):
                class_idx = i // modes_per_class
                mode_to_class[mode_idx] = min(class_idx, num_precision_classes - 1)
        
        class_to_mode = {v: available[k].mode for k, v in mode_to_class.items()}
        
        # Convert assignments to class indices
        class_assignments = torch.tensor(
            [mode_to_class[a.item()] for a in best_assignments],
            device=device, dtype=torch.long
        )
        
        return class_assignments, class_to_mode


def compute_activation_importance(
    weight: torch.Tensor,
    activation_squared_mean: torch.Tensor,
    block_size: int,
    num_groups: int,
) -> torch.Tensor:
    """
    Compute activation-aware importance for each group.
    
    I = E[a^2] * ||w||^2
    
    Args:
        weight: Weight tensor of shape (out_features, in_features).
        activation_squared_mean: E[a^2] for each input feature.
        block_size: Block dimension.
        num_groups: Number of groups per block.
        
    Returns:
        importances: Importance tensor of shape (num_blocks * num_groups,).
    """
    from pgrq.core.blocking import reshape_to_blocks, split_into_groups
    
    out_features, in_features = weight.shape
    
    # Reshape weight to blocks
    blocks, _ = reshape_to_blocks(weight, block_size)
    N_blocks = blocks.shape[0]
    
    # Split into groups
    groups = split_into_groups(blocks, num_groups)  # (N_blocks, G, gd)
    
    # Compute squared norm of each group
    group_norms_sq = (groups ** 2).sum(dim=-1)  # (N_blocks, G)
    
    # Expand activation importance to match weight structure
    # This is approximate - we average activation importance over the block
    act_sq = activation_squared_mean.flatten()
    
    # Pad activation importance to match weight size
    total_in = in_features
    if len(act_sq) < total_in:
        act_sq = torch.cat([
            act_sq,
            torch.zeros(total_in - len(act_sq), device=act_sq.device)
        ])
    elif len(act_sq) > total_in:
        act_sq = act_sq[:total_in]
    
    # Tile activation importance to cover all output features
    act_tiled = act_sq.repeat(out_features)
    
    # Pad to match blocks
    padded_size = N_blocks * block_size
    if len(act_tiled) < padded_size:
        act_tiled = torch.cat([
            act_tiled,
            torch.zeros(padded_size - len(act_tiled), device=act_tiled.device)
        ])
    
    # Reshape to blocks and groups
    act_blocks = act_tiled[:padded_size].view(N_blocks, block_size)
    act_groups = act_blocks.view(N_blocks, num_groups, -1)  # (N_blocks, G, gd)
    
    # Average activation importance per group
    act_importance = act_groups.mean(dim=-1)  # (N_blocks, G)
    
    # Final importance: E[a^2] * ||w||^2
    importances = act_importance * group_norms_sq  # (N_blocks, G)
    
    return importances.flatten()  # (N_blocks * G,)


def compute_raw_importance(
    weight: torch.Tensor,
    block_size: int,
    num_groups: int,
) -> torch.Tensor:
    """
    Compute raw magnitude-based importance (no activation info).
    
    I = ||w||^2
    
    Args:
        weight: Weight tensor.
        block_size: Block dimension.
        num_groups: Number of groups per block.
        
    Returns:
        importances: Importance tensor of shape (num_blocks * num_groups,).
    """
    from pgrq.core.blocking import reshape_to_blocks, split_into_groups
    
    # Reshape weight to blocks
    blocks, _ = reshape_to_blocks(weight, block_size)
    
    # Split into groups
    groups = split_into_groups(blocks, num_groups)  # (N_blocks, G, gd)
    
    # Compute squared norm of each group
    importances = (groups ** 2).sum(dim=-1)  # (N_blocks, G)
    
    return importances.flatten()
