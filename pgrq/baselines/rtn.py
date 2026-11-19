"""
Round-to-Nearest (RTN) baseline quantizer.

Simple per-channel or per-group uniform quantization.
"""

import torch
from typing import Dict, Tuple, Optional
import math


class RTNQuantizer:
    """
    Round-to-Nearest (RTN) quantizer.
    
    Implements simple uniform quantization as a baseline.
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize RTN quantizer.
        
        Args:
            bits: Bits per weight (2, 3, 4, 8).
            group_size: Size of quantization groups (0 for per-channel).
            symmetric: Use symmetric quantization.
            device: Target device.
        """
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.device = device
        
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
    
    def quantize(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a weight tensor.
        
        Args:
            weight: Weight tensor of shape (out_features, in_features).
            
        Returns:
            quantized: Quantized int tensor.
            scales: Scale factors.
            zeros: Zero points (None if symmetric).
        """
        weight = weight.to(self.device)
        out_features, in_features = weight.shape
        
        if self.group_size > 0:
            # Group-wise quantization
            # Pad if needed
            if in_features % self.group_size != 0:
                pad_size = self.group_size - (in_features % self.group_size)
                weight = torch.nn.functional.pad(weight, (0, pad_size))
                in_features = weight.shape[1]
            
            num_groups = in_features // self.group_size
            weight = weight.view(out_features, num_groups, self.group_size)
            
            if self.symmetric:
                # Symmetric: scale = max(|w|) / qmax
                max_abs = weight.abs().max(dim=-1, keepdim=True).values
                scales = max_abs / self.qmax
                scales = scales.clamp(min=1e-10)
                zeros = None
                
                quantized = (weight / scales).round().clamp(self.qmin, self.qmax).to(torch.int8)
            else:
                # Asymmetric: scale = (max - min) / (qmax - qmin)
                wmin = weight.min(dim=-1, keepdim=True).values
                wmax = weight.max(dim=-1, keepdim=True).values
                
                scales = (wmax - wmin) / (self.qmax - self.qmin)
                scales = scales.clamp(min=1e-10)
                zeros = (-wmin / scales).round().clamp(0, self.qmax - self.qmin)
                
                quantized = ((weight / scales) + zeros).round().clamp(0, self.qmax - self.qmin).to(torch.uint8)
            
            # Reshape back
            quantized = quantized.view(out_features, in_features)
            scales = scales.squeeze(-1)
            if zeros is not None:
                zeros = zeros.squeeze(-1)
        else:
            # Per-channel quantization
            if self.symmetric:
                max_abs = weight.abs().max(dim=1, keepdim=True).values
                scales = max_abs / self.qmax
                scales = scales.clamp(min=1e-10)
                zeros = None
                
                quantized = (weight / scales).round().clamp(self.qmin, self.qmax).to(torch.int8)
            else:
                wmin = weight.min(dim=1, keepdim=True).values
                wmax = weight.max(dim=1, keepdim=True).values
                
                scales = (wmax - wmin) / (self.qmax - self.qmin)
                scales = scales.clamp(min=1e-10)
                zeros = (-wmin / scales).round()
                
                quantized = ((weight / scales) + zeros).round().clamp(0, self.qmax - self.qmin).to(torch.uint8)
        
        return quantized, scales, zeros
    
    def dequantize(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Dequantize a tensor.
        
        Args:
            quantized: Quantized int tensor.
            scales: Scale factors.
            zeros: Zero points (None if symmetric).
            original_shape: Original weight shape (for unpadding).
            
        Returns:
            weight: Dequantized weight tensor.
        """
        quantized = quantized.float()
        
        if self.group_size > 0:
            out_features, in_features = quantized.shape
            num_groups = scales.shape[1]
            
            quantized = quantized.view(out_features, num_groups, self.group_size)
            scales = scales.unsqueeze(-1)
            
            if zeros is not None:
                zeros = zeros.unsqueeze(-1)
                weight = (quantized - zeros) * scales
            else:
                weight = quantized * scales
            
            weight = weight.view(out_features, in_features)
        else:
            if zeros is not None:
                weight = (quantized - zeros) * scales
            else:
                weight = quantized * scales
        
        # Unpad if needed
        if original_shape is not None:
            weight = weight[:original_shape[0], :original_shape[1]]
        
        return weight
    
    def get_compressed_size(
        self,
        weight_shape: Tuple[int, int],
    ) -> Dict[str, int]:
        """
        Compute compressed size in bytes.
        
        Args:
            weight_shape: (out_features, in_features).
            
        Returns:
            sizes: Dict with size breakdown.
        """
        out_features, in_features = weight_shape
        
        # Quantized weights
        weight_bits = out_features * in_features * self.bits
        weight_bytes = math.ceil(weight_bits / 8)
        
        # Scales (fp16)
        if self.group_size > 0:
            num_groups = math.ceil(in_features / self.group_size)
            scale_bytes = out_features * num_groups * 2
        else:
            scale_bytes = out_features * 2
        
        # Zeros (if asymmetric)
        if not self.symmetric:
            zero_bytes = scale_bytes  # Same shape as scales
        else:
            zero_bytes = 0
        
        total_bytes = weight_bytes + scale_bytes + zero_bytes
        bits_per_weight = total_bytes * 8 / (out_features * in_features)
        
        return {
            "weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "zero_bytes": zero_bytes,
            "total_bytes": total_bytes,
            "bits_per_weight": bits_per_weight,
        }


def quantize_model_rtn(
    model,
    bits: int = 4,
    group_size: int = 128,
    device: str = "auto",
    verbose: bool = True,
) -> Dict:
    """
    Quantize a HuggingFace model using RTN.
    
    Args:
        model: HuggingFace model.
        bits: Bits per weight.
        group_size: Quantization group size.
        device: Target device.
        verbose: Print progress.
        
    Returns:
        stats: Quantization statistics.
    """
    import torch
    from tqdm import tqdm
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    quantizer = RTNQuantizer(bits=bits, group_size=group_size, device=device)
    
    total_original = 0
    total_compressed = 0
    
    for name, module in tqdm(model.named_modules(), desc="Quantizing RTN", disable=not verbose):
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            original_shape = weight.shape
            
            # Quantize
            quantized, scales, zeros = quantizer.quantize(weight)
            
            # Dequantize and replace
            dequantized = quantizer.dequantize(quantized, scales, zeros, original_shape)
            module.weight.data = dequantized.to(weight.dtype)
            
            # Stats
            sizes = quantizer.get_compressed_size(original_shape)
            total_original += original_shape[0] * original_shape[1] * 2  # fp16
            total_compressed += sizes["total_bytes"]
    
    stats = {
        "original_bytes": total_original,
        "compressed_bytes": total_compressed,
        "compression_ratio": total_original / total_compressed,
        "bits_per_weight": total_compressed * 8 / (total_original // 2),
    }
    
    if verbose:
        print(f"RTN {bits}-bit quantization complete")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Bits per weight: {stats['bits_per_weight']:.3f}")
    
    return stats
