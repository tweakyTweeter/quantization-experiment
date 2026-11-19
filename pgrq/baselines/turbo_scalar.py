"""
TurboQuant-style scalar quantization baseline.

Implements Beta-Max-Lloyd scalar quantization as a standalone baseline
(without the full PGR-Q pipeline) for comparison.
"""

import torch
from typing import Dict, Tuple, Optional
import math

from pgrq.core.blocking import reshape_to_blocks, reshape_from_blocks, extract_norms, apply_norms
from pgrq.core.fwht import fwht_rotate, fwht_rotate_inverse
from pgrq.codebooks.scalar import BetaMaxLloydQuantizer


class TurboScalarQuantizer:
    """
    TurboQuant-style scalar quantization.
    
    Uses:
    1. Block reshaping
    2. Norm separation
    3. Hadamard rotation
    4. Beta-Max-Lloyd scalar quantization per coordinate
    
    This is a simpler baseline that doesn't use learned codebooks.
    """
    
    def __init__(
        self,
        bits: int = 4,
        block_size: int = 64,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        Initialize TurboScalar quantizer.
        
        Args:
            bits: Bits per coordinate.
            block_size: Block dimension (must be power of 2).
            device: Target device.
            seed: Random seed for Hadamard signs.
        """
        self.bits = bits
        self.block_size = block_size
        self.device = device
        self.seed = seed
        
        # Initialize scalar quantizer
        self.scalar_quantizer = BetaMaxLloydQuantizer(
            bits=bits,
            block_size=block_size,
            device=device,
            seed=seed,
        )
    
    def quantize(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize a weight tensor.
        
        Args:
            weight: Weight tensor of shape (out_features, in_features).
            
        Returns:
            codes: Quantized codes of shape (num_blocks, block_size).
            norms: Block norms of shape (num_blocks, 1).
            shape_info: Shape info for reconstruction.
        """
        weight = weight.to(self.device)
        
        # Reshape to blocks
        blocks, shape_info = reshape_to_blocks(weight, self.block_size)
        
        # Extract norms
        unit_dirs, norms = extract_norms(blocks)
        
        # Hadamard rotation
        rotated = fwht_rotate(unit_dirs, self.seed)
        
        # Scalar quantization
        codes, _ = self.scalar_quantizer.quantize(rotated.flatten())
        codes = codes.view(rotated.shape)
        
        return codes, norms, shape_info
    
    def dequantize(
        self,
        codes: torch.Tensor,
        norms: torch.Tensor,
        shape_info: Dict,
    ) -> torch.Tensor:
        """
        Dequantize codes back to weight tensor.
        
        Args:
            codes: Quantized codes.
            norms: Block norms.
            shape_info: Shape info from quantization.
            
        Returns:
            weight: Reconstructed weight tensor.
        """
        # Decode scalars
        rotated_recon = self.scalar_quantizer.decode(codes.flatten())
        rotated_recon = rotated_recon.view(codes.shape)
        
        # Inverse Hadamard
        unit_dirs_recon = fwht_rotate_inverse(rotated_recon, self.seed)
        
        # Apply norms
        blocks_recon = apply_norms(unit_dirs_recon, norms)
        
        # Reshape back
        weight_recon = reshape_from_blocks(blocks_recon, shape_info)
        
        return weight_recon
    
    def get_compressed_size(
        self,
        weight_shape: Tuple[int, int],
    ) -> Dict[str, float]:
        """
        Compute compressed size.
        
        Args:
            weight_shape: Original weight shape.
            
        Returns:
            sizes: Dict with size breakdown.
        """
        out_features, in_features = weight_shape
        total_elements = out_features * in_features
        
        # Pad to block size
        padded_elements = math.ceil(total_elements / self.block_size) * self.block_size
        num_blocks = padded_elements // self.block_size
        
        # Codes: bits per element
        code_bits = padded_elements * self.bits
        code_bytes = math.ceil(code_bits / 8)
        
        # Norms: fp16 per block
        norm_bytes = num_blocks * 2
        
        total_bytes = code_bytes + norm_bytes
        bits_per_weight = total_bytes * 8 / total_elements
        
        return {
            "code_bytes": code_bytes,
            "norm_bytes": norm_bytes,
            "total_bytes": total_bytes,
            "bits_per_weight": bits_per_weight,
        }


def quantize_model_turbo_scalar(
    model,
    bits: int = 4,
    block_size: int = 64,
    device: str = "auto",
    verbose: bool = True,
) -> Dict:
    """
    Quantize a HuggingFace model using TurboQuant scalar quantization.
    
    Args:
        model: HuggingFace model.
        bits: Bits per coordinate.
        block_size: Block size.
        device: Target device.
        verbose: Print progress.
        
    Returns:
        stats: Quantization statistics.
    """
    from tqdm import tqdm
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    quantizer = TurboScalarQuantizer(bits=bits, block_size=block_size, device=device)
    
    total_original = 0
    total_compressed = 0
    
    for name, module in tqdm(model.named_modules(), desc="Quantizing TurboScalar", disable=not verbose):
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            original_shape = weight.shape
            
            # Quantize
            codes, norms, shape_info = quantizer.quantize(weight)
            
            # Dequantize and replace
            dequantized = quantizer.dequantize(codes, norms, shape_info)
            module.weight.data = dequantized.to(weight.dtype)
            
            # Stats
            sizes = quantizer.get_compressed_size(original_shape)
            total_original += original_shape[0] * original_shape[1] * 2  # fp16
            total_compressed += sizes["total_bytes"]
    
    stats = {
        "method": "turbo_scalar",
        "bits": bits,
        "block_size": block_size,
        "original_bytes": total_original,
        "compressed_bytes": total_compressed,
        "compression_ratio": total_original / total_compressed,
        "bits_per_weight": total_compressed * 8 / (total_original // 2),
    }
    
    if verbose:
        print(f"TurboScalar {bits}-bit quantization complete")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Bits per weight: {stats['bits_per_weight']:.3f}")
    
    return stats
