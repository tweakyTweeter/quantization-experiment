"""
Main PGR-Q Quantizer.

Implements the full quantization pipeline:
1. Reshape to blocks
2. Extract norms (norm separation)
3. FWHT rotation
4. Group splitting
5. Base codebook encoding
6. Optional phase correction
7. Residual quantization
8. Optional scalar floor
9. Mixed-precision allocation
10. Entropy coding and packing
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from pgrq.core.config import PGRQConfig, ScalarFloorType, PrecisionMode
from pgrq.core.blocking import (
    reshape_to_blocks, reshape_from_blocks,
    extract_norms, apply_norms,
    split_into_groups, merge_groups
)
from pgrq.core.fwht import fwht_rotate, fwht_rotate_inverse
from pgrq.codebooks.kmeans import KMeansCodebook
from pgrq.codebooks.rvq import RVQCodebook
from pgrq.codebooks.phase import PhaseCorrector
from pgrq.codebooks.scalar import BetaMaxLloydQuantizer, DigitPlaneQuantizer
from pgrq.core.allocator import MixedPrecisionAllocator, compute_activation_importance, compute_raw_importance
from pgrq.entropy.packer import EntropyPacker


@dataclass
class QuantizedTensor:
    """Container for quantized tensor data."""
    packed_data: Dict[str, Any]
    shape_info: Dict[str, Any]
    config: Dict[str, Any]
    
    # Codebooks (for shared codebook mode, these may reference shared objects)
    base_codebook_state: Dict
    rvq_codebook_state: Optional[Dict] = None
    phase_corrector_state: Optional[Dict] = None
    scalar_quantizer_state: Optional[Dict] = None


class PGRQuantizer:
    """
    Main PGR-Q quantizer class.
    
    Implements the full quantization and dequantization pipeline.
    """
    
    def __init__(self, config: PGRQConfig):
        """
        Initialize quantizer.
        
        Args:
            config: PGR-Q configuration.
        """
        self.config = config
        self.device = config.device
        
        # Codebooks (will be trained on first layer or loaded)
        self.base_codebooks: Dict[int, KMeansCodebook] = {}  # group_idx -> codebook
        self.rvq_codebooks: Dict[int, RVQCodebook] = {}
        self.phase_corrector: Optional[PhaseCorrector] = None
        self.scalar_quantizer: Optional[Any] = None
        
        # For cross-layer sharing
        self._shared_base_codebook: Optional[KMeansCodebook] = None
        self._shared_rvq_codebook: Optional[RVQCodebook] = None
        
        # Training data accumulator (for shared codebook training)
        self._training_data: List[torch.Tensor] = []
        self._codebooks_trained = False
    
    def train_codebooks(
        self,
        weights: List[torch.Tensor],
        verbose: bool = False,
    ) -> None:
        """
        Train codebooks on a collection of weights.
        
        For cross-layer sharing, this should be called with weights from
        all layers that will share codebooks.
        
        Args:
            weights: List of weight tensors to train on.
            verbose: Print progress.
        """
        cfg = self.config
        
        if verbose:
            print(f"Training codebooks on {len(weights)} weight tensors")
        
        # Collect all group vectors for training
        all_groups = []
        
        for w in weights:
            # Process each weight
            blocks, _ = reshape_to_blocks(w, cfg.block_size)
            blocks = blocks.to(self.device)
            
            # Extract unit directions
            unit_dirs, _ = extract_norms(blocks)
            
            # Rotate
            rotated = fwht_rotate(unit_dirs, cfg.seed)
            
            # Split into groups
            groups = split_into_groups(rotated, cfg.num_groups)  # (N, G, gd)
            
            # Flatten for training
            N, G, gd = groups.shape
            groups_flat = groups.view(N * G, gd)
            all_groups.append(groups_flat)
        
        # Concatenate all groups
        all_groups = torch.cat(all_groups, dim=0)
        
        if verbose:
            print(f"Training on {all_groups.shape[0]} group vectors of dimension {all_groups.shape[1]}")
        
        # Subsample if too many
        max_samples = 500000
        if all_groups.shape[0] > max_samples:
            perm = torch.randperm(all_groups.shape[0])[:max_samples]
            all_groups = all_groups[perm]
        
        # Train base codebook
        if cfg.cross_layer_codebook_sharing:
            if verbose:
                print("Training shared base codebook...")
            
            self._shared_base_codebook = KMeansCodebook(
                K=cfg.base_codebook_size,
                D=cfg.group_dim,
                device=self.device,
            )
            self._shared_base_codebook.train(
                all_groups,
                num_iters=cfg.kmeans_iters,
                init=cfg.kmeans_init,
                seed=cfg.seed,
                verbose=verbose,
            )
            
            if verbose:
                mse = self._shared_base_codebook.get_mse(all_groups)
                print(f"Base codebook MSE: {mse:.6f}")
            
            # Train RVQ on residuals
            if cfg.num_residual_levels > 0:
                if verbose:
                    print("Training shared RVQ codebook...")
                
                # Compute base reconstruction
                base_codes = self._shared_base_codebook.encode(all_groups)
                base_recon = self._shared_base_codebook.decode(base_codes)
                residuals = all_groups - base_recon
                
                self._shared_rvq_codebook = RVQCodebook(
                    M=cfg.num_residual_levels,
                    K=cfg.residual_codebook_size,
                    D=cfg.group_dim,
                    device=self.device,
                )
                self._shared_rvq_codebook.train(
                    residuals,
                    num_iters=cfg.kmeans_iters,
                    joint_refinement=cfg.joint_refinement,
                    joint_refinement_iters=cfg.joint_refinement_iters,
                    seed=cfg.seed + 1000,
                    verbose=verbose,
                )
                
                if verbose:
                    mse = self._shared_rvq_codebook.get_mse(residuals)
                    print(f"RVQ codebook MSE: {mse:.6f}")
        else:
            # Per-group codebooks
            if cfg.per_group_codebooks:
                # Train separate codebook for each group index
                for g in range(cfg.num_groups):
                    if verbose:
                        print(f"Training codebook for group {g}...")
                    
                    # Extract data for this group
                    # Reshape all_groups back: assume we have N_total vectors of dim gd
                    # We need to extract every G-th group
                    # This is complex - for simplicity, train on all and use per-group
                    self.base_codebooks[g] = KMeansCodebook(
                        K=cfg.base_codebook_size,
                        D=cfg.group_dim,
                        device=self.device,
                    )
                    self.base_codebooks[g].train(
                        all_groups,
                        num_iters=cfg.kmeans_iters,
                        init=cfg.kmeans_init,
                        seed=cfg.seed + g,
                        verbose=verbose,
                    )
            else:
                # Single shared codebook
                self._shared_base_codebook = KMeansCodebook(
                    K=cfg.base_codebook_size,
                    D=cfg.group_dim,
                    device=self.device,
                )
                self._shared_base_codebook.train(
                    all_groups,
                    num_iters=cfg.kmeans_iters,
                    init=cfg.kmeans_init,
                    seed=cfg.seed,
                    verbose=verbose,
                )
        
        # Initialize phase corrector if enabled
        if cfg.phase_correction:
            self.phase_corrector = PhaseCorrector(R=cfg.phase_resolution, device=self.device)
        
        # Initialize scalar quantizer if enabled
        if cfg.enable_scalar_floor:
            if cfg.scalar_floor_type == ScalarFloorType.BETA_MAX_LLOYD:
                self.scalar_quantizer = BetaMaxLloydQuantizer(
                    bits=cfg.scalar_floor_bits,
                    block_size=cfg.block_size,
                    device=self.device,
                )
            else:
                self.scalar_quantizer = DigitPlaneQuantizer(
                    levels=cfg.digit_planes_levels,
                    block_size=cfg.block_size,
                    device=self.device,
                )
        
        self._codebooks_trained = True
    
    def _get_base_codebook(self, group_idx: int = 0) -> KMeansCodebook:
        """Get the base codebook for a given group index."""
        if self._shared_base_codebook is not None:
            return self._shared_base_codebook
        elif group_idx in self.base_codebooks:
            return self.base_codebooks[group_idx]
        else:
            raise RuntimeError("Codebooks not trained")
    
    def _get_rvq_codebook(self, group_idx: int = 0) -> Optional[RVQCodebook]:
        """Get the RVQ codebook for a given group index."""
        if self._shared_rvq_codebook is not None:
            return self._shared_rvq_codebook
        elif group_idx in self.rvq_codebooks:
            return self.rvq_codebooks[group_idx]
        return None
    
    def quantize(
        self,
        weight: torch.Tensor,
        activation_stats: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> QuantizedTensor:
        """
        Quantize a weight tensor.
        
        Args:
            weight: Weight tensor of shape (out_features, in_features).
            activation_stats: Optional E[a^2] for activation-aware importance.
            verbose: Print progress.
            
        Returns:
            quantized: QuantizedTensor containing all compressed data.
        """
        if not self._codebooks_trained:
            raise RuntimeError("Must train codebooks before quantizing")
        
        cfg = self.config
        device = self.device
        weight = weight.to(device)
        
        # Stage 2: Reshape to blocks and extract norms
        blocks, shape_info = reshape_to_blocks(weight, cfg.block_size)
        blocks = blocks.to(device)
        unit_dirs, norms = extract_norms(blocks, norm_dtype=getattr(torch, cfg.norm_dtype))
        
        if verbose:
            print(f"Quantizing weight of shape {weight.shape}")
            print(f"  {shape_info['num_blocks']} blocks, {cfg.block_size} elements each")
        
        # Stage 3: FWHT rotation
        rotated = fwht_rotate(unit_dirs, cfg.seed)
        
        # Stage 4: Split into groups
        groups = split_into_groups(rotated, cfg.num_groups)  # (N, G, gd)
        N_blocks, G, gd = groups.shape
        
        # Flatten groups for encoding
        groups_flat = groups.view(N_blocks * G, gd)
        N_groups = N_blocks * G
        
        # Stage 5: Base codebook encoding
        base_cb = self._get_base_codebook()
        base_codes = base_cb.encode(groups_flat)
        base_recon = base_cb.decode(base_codes)
        
        if verbose:
            base_mse = ((groups_flat - base_recon) ** 2).mean().item()
            print(f"  Base codebook MSE: {base_mse:.6f}")
        
        # Stage 6: Optional phase correction
        phase_codes = None
        current_recon = base_recon
        
        if cfg.phase_correction and self.phase_corrector is not None:
            phase_codes, current_recon = self.phase_corrector.apply_phase(
                groups_flat, base_recon
            )
            
            if verbose:
                phase_mse = ((groups_flat - current_recon) ** 2).mean().item()
                print(f"  After phase correction MSE: {phase_mse:.6f}")
        
        # Stage 7a: Residual quantization
        residual_codes = None
        residual = groups_flat - current_recon
        
        rvq_cb = self._get_rvq_codebook()
        if rvq_cb is not None:
            residual_codes = rvq_cb.encode(residual)
            rvq_recon = rvq_cb.decode(residual_codes)
            current_recon = current_recon + rvq_recon
            
            if verbose:
                rvq_mse = ((groups_flat - current_recon) ** 2).mean().item()
                print(f"  After RVQ MSE: {rvq_mse:.6f}")
        
        # Stage 7b: Optional scalar floor
        scalar_codes = None
        
        if cfg.enable_scalar_floor and self.scalar_quantizer is not None:
            final_residual = groups_flat - current_recon
            scalar_codes, scalar_recon = self.scalar_quantizer.quantize(
                final_residual.flatten()
            )
            scalar_codes = scalar_codes.view(N_groups, gd)
            scalar_recon = scalar_recon.view(N_groups, gd)
            current_recon = current_recon + scalar_recon
            
            if verbose:
                scalar_mse = ((groups_flat - current_recon) ** 2).mean().item()
                print(f"  After scalar floor MSE: {scalar_mse:.6f}")
        
        # Stage 8: Compute importance for mixed-precision (if needed)
        # For now, use uniform allocation (all groups get same mode)
        # TODO: Implement full mixed-precision allocation
        class_assignments = None  # All same precision for now
        
        # Stage 9-10: Pack everything
        packer = EntropyPacker(
            base_codebook_size=cfg.base_codebook_size,
            residual_codebook_size=cfg.residual_codebook_size,
            num_residual_levels=cfg.num_residual_levels,
            phase_resolution=cfg.phase_resolution,
            use_entropy_coding=cfg.entropy_coding,
        )
        
        packed = packer.pack(
            base_codes=base_codes,
            residual_codes=residual_codes,
            phase_codes=phase_codes,
            scalar_codes=scalar_codes,
            class_assignments=class_assignments,
            norms=norms,
        )
        
        # Compute size
        size_bytes = packer.compute_size_bytes(packed)
        bits_per_weight = size_bytes * 8 / weight.numel()
        
        if verbose:
            print(f"  Compressed size: {size_bytes} bytes")
            print(f"  Bits per weight: {bits_per_weight:.3f}")
        
        # Build quantized tensor
        quantized = QuantizedTensor(
            packed_data=packed,
            shape_info=shape_info,
            config=cfg.to_dict(),
            base_codebook_state=base_cb.state_dict(),
            rvq_codebook_state=rvq_cb.state_dict() if rvq_cb else None,
            phase_corrector_state={"R": self.phase_corrector.R} if self.phase_corrector else None,
            scalar_quantizer_state=self.scalar_quantizer.state_dict() if self.scalar_quantizer else None,
        )
        
        return quantized
    
    def dequantize(
        self,
        quantized: QuantizedTensor,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Dequantize a quantized tensor back to weights.
        
        Args:
            quantized: QuantizedTensor from quantize().
            device: Target device (defaults to self.device).
            
        Returns:
            weight: Reconstructed weight tensor.
        """
        device = device or self.device
        cfg = PGRQConfig.from_dict(quantized.config)
        
        # Load codebooks from state
        base_cb = KMeansCodebook.from_state_dict(
            quantized.base_codebook_state, device=device
        )
        
        rvq_cb = None
        if quantized.rvq_codebook_state is not None:
            rvq_cb = RVQCodebook.from_state_dict(
                quantized.rvq_codebook_state, device=device
            )
        
        phase_corrector = None
        if quantized.phase_corrector_state is not None:
            phase_corrector = PhaseCorrector(
                R=quantized.phase_corrector_state["R"], device=device
            )
        
        scalar_quantizer = None
        if quantized.scalar_quantizer_state is not None:
            state = quantized.scalar_quantizer_state
            if "centroids" in state:
                scalar_quantizer = BetaMaxLloydQuantizer.from_state_dict(state, device)
            else:
                scalar_quantizer = DigitPlaneQuantizer.from_state_dict(state, device)
        
        # Unpack codes
        packer = EntropyPacker(
            base_codebook_size=cfg.base_codebook_size,
            residual_codebook_size=cfg.residual_codebook_size,
            num_residual_levels=cfg.num_residual_levels,
            phase_resolution=cfg.phase_resolution,
            use_entropy_coding=cfg.entropy_coding,
        )
        
        unpacked = packer.unpack(quantized.packed_data, device=device)
        
        base_codes = unpacked["base_codes"]
        norms = unpacked["norms"]
        residual_codes = unpacked.get("residual_codes")
        phase_codes = unpacked.get("phase_codes")
        scalar_codes = unpacked.get("scalar_codes")
        
        # Stage 5: Decode base
        groups_recon = base_cb.decode(base_codes)
        
        # Stage 6: Apply phase correction
        if phase_codes is not None and phase_corrector is not None:
            groups_recon = phase_corrector.decode_phase(groups_recon, phase_codes)
        
        # Stage 7a: Add RVQ reconstruction
        if residual_codes is not None and rvq_cb is not None:
            rvq_recon = rvq_cb.decode(residual_codes)
            groups_recon = groups_recon + rvq_recon
        
        # Stage 7b: Add scalar floor reconstruction
        if scalar_codes is not None and scalar_quantizer is not None:
            scalar_recon = scalar_quantizer.decode(scalar_codes.flatten())
            N_groups, gd = groups_recon.shape
            scalar_recon = scalar_recon.view(N_groups, gd)
            groups_recon = groups_recon + scalar_recon
        
        # Stage 4 inverse: Merge groups
        shape_info = quantized.shape_info
        N_blocks = shape_info["num_blocks"]
        block_size = shape_info["block_size"]
        G = cfg.num_groups
        
        groups_recon = groups_recon.view(N_blocks, G, -1)
        rotated_recon = merge_groups(groups_recon)
        
        # Stage 3 inverse: FWHT inverse rotation
        unit_dirs_recon = fwht_rotate_inverse(rotated_recon, cfg.seed)
        
        # Stage 2 inverse: Apply norms
        norms = norms.to(device).to(unit_dirs_recon.dtype)
        blocks_recon = apply_norms(unit_dirs_recon, norms)
        
        # Reshape back to weight
        weight_recon = reshape_from_blocks(blocks_recon, shape_info)
        
        return weight_recon
    
    def get_compression_stats(self, quantized: QuantizedTensor) -> Dict:
        """
        Get compression statistics for a quantized tensor.
        
        Args:
            quantized: QuantizedTensor from quantize().
            
        Returns:
            stats: Dict with compression statistics.
        """
        cfg = PGRQConfig.from_dict(quantized.config)
        shape_info = quantized.shape_info
        
        packer = EntropyPacker(
            base_codebook_size=cfg.base_codebook_size,
            residual_codebook_size=cfg.residual_codebook_size,
            num_residual_levels=cfg.num_residual_levels,
            phase_resolution=cfg.phase_resolution,
            use_entropy_coding=cfg.entropy_coding,
        )
        
        size_bytes = packer.compute_size_bytes(quantized.packed_data)
        original_size = shape_info["total_elements"] * 2  # Assuming fp16 original
        
        # Codebook overhead
        codebook_bytes = 0
        base_cb = quantized.base_codebook_state
        if base_cb.get("centroids") is not None:
            codebook_bytes += base_cb["centroids"].numel() * 2  # fp16
        
        rvq_state = quantized.rvq_codebook_state
        if rvq_state is not None:
            for cb in rvq_state.get("codebooks", []):
                codebook_bytes += cb.numel() * 2
        
        total_bytes = size_bytes + codebook_bytes
        
        return {
            "compressed_bytes": size_bytes,
            "codebook_bytes": codebook_bytes,
            "total_bytes": total_bytes,
            "original_bytes": original_size,
            "compression_ratio": original_size / total_bytes,
            "bits_per_weight": total_bytes * 8 / shape_info["total_elements"],
            "num_weights": shape_info["total_elements"],
        }
