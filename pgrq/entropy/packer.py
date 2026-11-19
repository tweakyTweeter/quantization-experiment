"""
Stage 10: Packing utilities.

Two packing methods:
1. Entropy coding (DEFAULT): Use range coder for each symbol stream
2. Mixed-radix packing (ALTERNATIVE): Pack integers using variable bases

The two should NOT be used together.
"""

import torch
from typing import List, Tuple, Dict, Optional
import struct
import math


class MixedRadixPacker:
    """
    Mixed-radix integer packing.
    
    Packs multiple integers with different ranges into a single big integer:
    Z = a0 + K0*a1 + K0*K1*a2 + ...
    
    This is the ALTERNATIVE to entropy coding.
    """
    
    def __init__(self, radices: List[int]):
        """
        Initialize packer with radices (max values + 1 for each field).
        
        Args:
            radices: List of radices [K0, K1, K2, ...].
        """
        self.radices = radices
        self.num_fields = len(radices)
        
        # Precompute cumulative products for packing
        self.cum_products = [1]
        for r in radices[:-1]:
            self.cum_products.append(self.cum_products[-1] * r)
        
        # Total range
        self.total_range = self.cum_products[-1] * radices[-1] if radices else 1
    
    def pack(self, values: List[torch.Tensor]) -> torch.Tensor:
        """
        Pack multiple value tensors into a single integer tensor.
        
        Args:
            values: List of tensors, one per field.
            
        Returns:
            packed: Single tensor with packed values.
        """
        if len(values) != self.num_fields:
            raise ValueError(f"Expected {self.num_fields} fields, got {len(values)}")
        
        result = torch.zeros_like(values[0], dtype=torch.long)
        
        for i, (v, cp) in enumerate(zip(values, self.cum_products)):
            result += v.long() * cp
        
        return result
    
    def unpack(self, packed: torch.Tensor) -> List[torch.Tensor]:
        """
        Unpack a single integer tensor into multiple value tensors.
        
        Args:
            packed: Packed integer tensor.
            
        Returns:
            values: List of unpacked tensors.
        """
        values = []
        remaining = packed.clone()
        
        for i, r in enumerate(self.radices):
            values.append(remaining % r)
            remaining = remaining // r
        
        return values
    
    def bits_per_element(self) -> float:
        """Compute bits per element for this packing."""
        return math.log2(self.total_range) if self.total_range > 0 else 0


class EntropyPacker:
    """
    Entropy-coded packing for quantized codes.
    
    This is the DEFAULT packing method. Groups codes by precision class
    for SIMD-regular packing.
    """
    
    def __init__(
        self,
        base_codebook_size: int = 256,
        residual_codebook_size: int = 256,
        num_residual_levels: int = 2,
        phase_resolution: int = 8,
        num_precision_classes: int = 4,
        use_entropy_coding: bool = True,
    ):
        """
        Initialize packer.
        
        Args:
            base_codebook_size: K0 for base codebook.
            residual_codebook_size: K for residual codebooks.
            num_residual_levels: M for residual levels.
            phase_resolution: R for phase correction.
            num_precision_classes: Number of distinct precision classes.
            use_entropy_coding: If True, use entropy coding; else mixed-radix.
        """
        self.K0 = base_codebook_size
        self.K = residual_codebook_size
        self.M = num_residual_levels
        self.R = phase_resolution
        self.num_classes = num_precision_classes
        self.use_entropy_coding = use_entropy_coding
    
    def pack(
        self,
        base_codes: torch.Tensor,
        residual_codes: Optional[List[torch.Tensor]] = None,
        phase_codes: Optional[torch.Tensor] = None,
        scalar_codes: Optional[torch.Tensor] = None,
        class_assignments: Optional[torch.Tensor] = None,
        norms: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Pack all quantization codes.
        
        Args:
            base_codes: Base codebook codes of shape (N,).
            residual_codes: List of M residual code tensors.
            phase_codes: Phase correction codes of shape (N,).
            scalar_codes: Scalar floor codes of shape (N, gd).
            class_assignments: Precision class per group of shape (N,).
            norms: Block norms to store.
            
        Returns:
            packed: Dict with packed data and metadata.
        """
        if self.use_entropy_coding:
            return self._pack_entropy(
                base_codes, residual_codes, phase_codes, 
                scalar_codes, class_assignments, norms
            )
        else:
            return self._pack_mixed_radix(
                base_codes, residual_codes, phase_codes,
                scalar_codes, class_assignments, norms
            )
    
    def _pack_entropy(
        self,
        base_codes: torch.Tensor,
        residual_codes: Optional[List[torch.Tensor]],
        phase_codes: Optional[torch.Tensor],
        scalar_codes: Optional[torch.Tensor],
        class_assignments: Optional[torch.Tensor],
        norms: Optional[torch.Tensor],
    ) -> Dict:
        """Pack using entropy coding."""
        from pgrq.entropy.range_coder import SymbolStream
        
        stream = SymbolStream()
        
        # Add base codes
        stream.add("base", base_codes, self.K0)
        
        # Add residual codes
        if residual_codes is not None:
            for m, rc in enumerate(residual_codes):
                stream.add(f"resid_{m}", rc, self.K)
        
        # Add phase codes
        if phase_codes is not None:
            stream.add("phase", phase_codes, self.R)
        
        # Add scalar codes (if present)
        if scalar_codes is not None:
            # Scalar codes may have different alphabet size
            max_scalar = scalar_codes.max().item() + 1
            stream.add("scalar", scalar_codes.flatten(), max_scalar)
        
        # Add class assignments
        if class_assignments is not None:
            stream.add("class", class_assignments, self.num_classes)
        
        # Encode all streams
        data, metadata = stream.encode_all()
        
        packed = {
            "method": "entropy",
            "data": data,
            "metadata": metadata,
            "norms": norms.cpu() if norms is not None else None,
            "shapes": {
                "base": list(base_codes.shape),
                "norms": list(norms.shape) if norms is not None else None,
            },
        }
        
        if scalar_codes is not None:
            packed["shapes"]["scalar"] = list(scalar_codes.shape)
        
        return packed
    
    def _pack_mixed_radix(
        self,
        base_codes: torch.Tensor,
        residual_codes: Optional[List[torch.Tensor]],
        phase_codes: Optional[torch.Tensor],
        scalar_codes: Optional[torch.Tensor],
        class_assignments: Optional[torch.Tensor],
        norms: Optional[torch.Tensor],
    ) -> Dict:
        """Pack using mixed-radix integer packing."""
        # Build radices for mixed-radix packing
        radices = [self.K0]  # Base codes
        values = [base_codes]
        
        if residual_codes is not None:
            for rc in residual_codes:
                radices.append(self.K)
                values.append(rc)
        
        if phase_codes is not None:
            radices.append(self.R)
            values.append(phase_codes)
        
        packer = MixedRadixPacker(radices)
        packed_values = packer.pack(values)
        
        result = {
            "method": "mixed_radix",
            "packed_values": packed_values.cpu(),
            "radices": radices,
            "norms": norms.cpu() if norms is not None else None,
            "shapes": {
                "base": list(base_codes.shape),
                "norms": list(norms.shape) if norms is not None else None,
            },
            "class_assignments": class_assignments.cpu() if class_assignments is not None else None,
        }
        
        # Handle scalar codes separately (too many values for mixed-radix)
        if scalar_codes is not None:
            result["scalar_codes"] = scalar_codes.cpu()
            result["shapes"]["scalar"] = list(scalar_codes.shape)
        
        return result
    
    def unpack(self, packed: Dict, device: str = "cpu") -> Dict:
        """
        Unpack quantization codes.
        
        Args:
            packed: Packed data from pack().
            device: Target device for tensors.
            
        Returns:
            unpacked: Dict with unpacked tensors.
        """
        if packed["method"] == "entropy":
            return self._unpack_entropy(packed, device)
        else:
            return self._unpack_mixed_radix(packed, device)
    
    def _unpack_entropy(self, packed: Dict, device: str) -> Dict:
        """Unpack entropy-coded data."""
        from pgrq.entropy.range_coder import SymbolStream
        
        streams = SymbolStream.decode_all(
            packed["data"],
            packed["metadata"],
            device=device,
        )
        
        result = {
            "base_codes": streams["base"],
            "norms": packed["norms"].to(device) if packed["norms"] is not None else None,
        }
        
        # Extract residual codes
        residual_codes = []
        m = 0
        while f"resid_{m}" in streams:
            residual_codes.append(streams[f"resid_{m}"])
            m += 1
        if residual_codes:
            result["residual_codes"] = residual_codes
        
        # Extract phase codes
        if "phase" in streams:
            result["phase_codes"] = streams["phase"]
        
        # Extract scalar codes
        if "scalar" in streams:
            shape = packed["shapes"]["scalar"]
            result["scalar_codes"] = streams["scalar"].view(shape)
        
        # Extract class assignments
        if "class" in streams:
            result["class_assignments"] = streams["class"]
        
        return result
    
    def _unpack_mixed_radix(self, packed: Dict, device: str) -> Dict:
        """Unpack mixed-radix packed data."""
        packer = MixedRadixPacker(packed["radices"])
        values = packer.unpack(packed["packed_values"].to(device))
        
        result = {
            "base_codes": values[0],
            "norms": packed["norms"].to(device) if packed["norms"] is not None else None,
        }
        
        # Extract residual codes
        if len(values) > 1:
            # Determine how many are residual vs phase
            num_resid = self.M if len(values) > 1 + self.M else len(values) - 1
            if packed.get("class_assignments") is not None:
                # Has phase codes
                num_resid = min(num_resid, len(values) - 2)
            
            result["residual_codes"] = values[1:1 + num_resid]
            
            if len(values) > 1 + num_resid:
                result["phase_codes"] = values[1 + num_resid]
        
        # Scalar codes stored separately
        if "scalar_codes" in packed:
            result["scalar_codes"] = packed["scalar_codes"].to(device)
        
        if packed.get("class_assignments") is not None:
            result["class_assignments"] = packed["class_assignments"].to(device)
        
        return result
    
    def compute_size_bytes(self, packed: Dict) -> int:
        """
        Compute total size in bytes of packed data.
        
        This is the MEASURED size, not theoretical.
        
        Args:
            packed: Packed data dict.
            
        Returns:
            size: Total bytes.
        """
        size = 0
        
        if packed["method"] == "entropy":
            size += len(packed["data"])
        else:
            # Mixed-radix: count bytes for packed values
            packed_values = packed["packed_values"]
            bits_per_val = math.log2(packed.get("radices", [256])[0])
            for r in packed.get("radices", [256])[1:]:
                bits_per_val += math.log2(r)
            bytes_per_val = math.ceil(bits_per_val / 8)
            size += packed_values.numel() * bytes_per_val
            
            if "scalar_codes" in packed:
                # Assume 1 byte per scalar code
                size += packed["scalar_codes"].numel()
        
        # Norms: stored in fp16 = 2 bytes each
        if packed.get("norms") is not None:
            size += packed["norms"].numel() * 2
        
        return size
