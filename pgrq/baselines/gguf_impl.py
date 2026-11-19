"""
GGUF-style quantization baseline.

Implements the quantization schemes used by llama.cpp/GGUF format:
- Q4_0: 4-bit with per-block (32 elements) absmax scaling, symmetric
- Q4_1: 4-bit with per-block min+scale (asymmetric)
- Q5_0: 5-bit symmetric
- Q5_1: 5-bit asymmetric
- Q8_0: 8-bit symmetric
- Q2_K: 2-bit with super-blocks (K-quant scheme)
- Q3_K: 3-bit K-quant
- Q4_K: 4-bit K-quant
- Q6_K: 6-bit K-quant

K-quants use a two-level block structure:
- Super-block of 256 elements
- Sub-blocks of 16-32 elements with local scales
- Scales themselves are quantized to fewer bits
"""

import torch
import math
from typing import Dict, Tuple


class GGUFQuantizer:
    """
    Implements GGUF quantization types.
    """
    
    def __init__(self, quant_type: str = "Q4_0"):
        self.quant_type = quant_type.upper()
    
    def quantize_tensor(self, W: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize a weight tensor using the specified GGUF type.
        
        Returns:
            W_rec: Reconstructed (dequantized) weight
            info: metadata including BPW
        """
        original_shape = W.shape
        W_flat = W.flatten().float()
        
        if self.quant_type == "Q4_0":
            W_rec, info = self._q4_0(W_flat)
        elif self.quant_type == "Q4_1":
            W_rec, info = self._q4_1(W_flat)
        elif self.quant_type == "Q5_0":
            W_rec, info = self._q5_0(W_flat)
        elif self.quant_type == "Q5_1":
            W_rec, info = self._q5_1(W_flat)
        elif self.quant_type == "Q8_0":
            W_rec, info = self._q8_0(W_flat)
        elif self.quant_type == "Q2_K":
            W_rec, info = self._qk(W_flat, bits=2)
        elif self.quant_type == "Q3_K":
            W_rec, info = self._qk(W_flat, bits=3)
        elif self.quant_type == "Q4_K":
            W_rec, info = self._qk(W_flat, bits=4)
        elif self.quant_type == "Q6_K":
            W_rec, info = self._qk(W_flat, bits=6)
        else:
            raise ValueError(f"Unknown quant type: {self.quant_type}")
        
        W_rec = W_rec.view(original_shape)
        info['mse'] = ((W.float() - W_rec) ** 2).mean().item()
        return W_rec, info
    
    def _q4_0(self, W: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Q4_0: 4-bit symmetric, block_size=32, fp16 scale per block."""
        block_size = 32
        n = W.numel()
        n_pad = math.ceil(n / block_size) * block_size
        
        W_pad = torch.zeros(n_pad)
        W_pad[:n] = W
        blocks = W_pad.view(-1, block_size)
        
        # Per-block absmax
        scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        
        # Quantize to [-8, 7] (4-bit signed)
        q = torch.clamp(torch.round(blocks / scales * 8), -8, 7)
        
        # Dequantize
        rec = q * scales / 8
        
        # BPW: 4 bits per element + 16 bits scale per 32 elements = 4 + 0.5 = 4.5
        bpw = 4 + 16 / block_size
        
        return rec.flatten()[:n], {'bpw': bpw, 'type': 'Q4_0'}
    
    def _q4_1(self, W: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Q4_1: 4-bit asymmetric, block_size=32, fp16 min+scale."""
        block_size = 32
        n = W.numel()
        n_pad = math.ceil(n / block_size) * block_size
        
        W_pad = torch.zeros(n_pad)
        W_pad[:n] = W
        blocks = W_pad.view(-1, block_size)
        
        mins = blocks.min(dim=1, keepdim=True).values
        maxs = blocks.max(dim=1, keepdim=True).values
        scales = (maxs - mins).clamp(min=1e-8) / 15  # [0, 15] for 4-bit unsigned
        
        q = torch.clamp(torch.round((blocks - mins) / scales), 0, 15)
        rec = q * scales + mins
        
        # BPW: 4 bits + (16+16)/32 = 4 + 1 = 5.0
        bpw = 4 + 32 / block_size
        
        return rec.flatten()[:n], {'bpw': bpw, 'type': 'Q4_1'}
    
    def _q5_0(self, W: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Q5_0: 5-bit symmetric, block_size=32."""
        block_size = 32
        n = W.numel()
        n_pad = math.ceil(n / block_size) * block_size
        
        W_pad = torch.zeros(n_pad)
        W_pad[:n] = W
        blocks = W_pad.view(-1, block_size)
        
        scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        q = torch.clamp(torch.round(blocks / scales * 16), -16, 15)
        rec = q * scales / 16
        
        bpw = 5 + 16 / block_size  # 5.5
        return rec.flatten()[:n], {'bpw': bpw, 'type': 'Q5_0'}
    
    def _q5_1(self, W: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Q5_1: 5-bit asymmetric, block_size=32."""
        block_size = 32
        n = W.numel()
        n_pad = math.ceil(n / block_size) * block_size
        
        W_pad = torch.zeros(n_pad)
        W_pad[:n] = W
        blocks = W_pad.view(-1, block_size)
        
        mins = blocks.min(dim=1, keepdim=True).values
        maxs = blocks.max(dim=1, keepdim=True).values
        scales = (maxs - mins).clamp(min=1e-8) / 31
        
        q = torch.clamp(torch.round((blocks - mins) / scales), 0, 31)
        rec = q * scales + mins
        
        bpw = 5 + 32 / block_size  # 6.0
        return rec.flatten()[:n], {'bpw': bpw, 'type': 'Q5_1'}
    
    def _q8_0(self, W: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Q8_0: 8-bit symmetric, block_size=32."""
        block_size = 32
        n = W.numel()
        n_pad = math.ceil(n / block_size) * block_size
        
        W_pad = torch.zeros(n_pad)
        W_pad[:n] = W
        blocks = W_pad.view(-1, block_size)
        
        scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        q = torch.clamp(torch.round(blocks / scales * 127), -127, 127)
        rec = q * scales / 127
        
        bpw = 8 + 16 / block_size  # 8.5
        return rec.flatten()[:n], {'bpw': bpw, 'type': 'Q8_0'}
    
    def _qk(self, W: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict]:
        """
        K-quant style: super-blocks of 256, sub-blocks of 16.
        Sub-block scales quantized to 6 bits relative to super-block scale.
        """
        super_block_size = 256
        sub_block_size = 16
        n_sub = super_block_size // sub_block_size  # 16 sub-blocks per super-block
        
        n = W.numel()
        n_pad = math.ceil(n / super_block_size) * super_block_size
        
        W_pad = torch.zeros(n_pad)
        W_pad[:n] = W
        
        # Process super-blocks
        n_super = n_pad // super_block_size
        W_blocks = W_pad.view(n_super, super_block_size)
        
        rec = torch.zeros_like(W_blocks)
        
        for sb in range(n_super):
            block = W_blocks[sb]
            
            # Super-block scale (fp16)
            super_scale = block.abs().max().clamp(min=1e-8)
            
            # Sub-block processing
            sub_blocks = block.view(n_sub, sub_block_size)
            sub_scales = sub_blocks.abs().max(dim=1).values.clamp(min=1e-8)
            
            # Quantize sub-scales to 6 bits relative to super_scale
            q_sub_scales = torch.clamp(torch.round(sub_scales / super_scale * 63), 0, 63)
            rec_sub_scales = q_sub_scales * super_scale / 63
            
            # Quantize values
            max_val = 2**(bits-1) - 1
            for i in range(n_sub):
                s = rec_sub_scales[i].clamp(min=1e-8)
                q = torch.clamp(torch.round(sub_blocks[i] / s * max_val), -max_val, max_val)
                rec[sb, i*sub_block_size:(i+1)*sub_block_size] = q * s / max_val
        
        # BPW calculation for K-quants:
        # Per super-block: 16 bits (super scale) + 16 bits (super min for asym, 0 for sym)
        #                  + n_sub * 6 bits (sub-block scales)
        #                  + super_block_size * bits (quantized values)
        overhead_per_super = 16 + n_sub * 6  # = 16 + 96 = 112 bits
        data_per_super = super_block_size * bits
        total_per_super = overhead_per_super + data_per_super
        bpw = total_per_super / super_block_size
        
        return rec.flatten()[:n], {'bpw': bpw, 'type': f'Q{bits}_K'}


def quantize_model_gguf(
    model,
    quant_type: str = "Q4_0",
    verbose: bool = True,
) -> Dict:
    """
    Quantize a HuggingFace model using GGUF-style quantization.
    
    Args:
        model: HuggingFace model
        quant_type: GGUF quantization type
        verbose: Print progress
        
    Returns:
        stats: Quantization statistics
    """
    import torch.nn as nn
    from tqdm import tqdm
    
    quantizer = GGUFQuantizer(quant_type)
    
    total_params = 0
    total_mse = 0.0
    bpw = None
    
    layers = [(n, m) for n, m in model.named_modules() 
              if isinstance(m, nn.Linear) and 'embed' not in n.lower() and 'lm_head' not in n.lower()]
    
    for name, module in tqdm(layers, desc=f"GGUF {quant_type}", disable=not verbose):
        W = module.weight.data
        W_rec, info = quantizer.quantize_tensor(W)
        module.weight.data = W_rec.to(W.dtype)
        
        total_params += W.numel()
        total_mse += info['mse'] * W.numel()
        bpw = info['bpw']
    
    stats = {
        'method': f'GGUF-{quant_type}',
        'quant_type': quant_type,
        'bpw': bpw,
        'mse': total_mse / total_params if total_params > 0 else 0,
        'total_params': total_params,
    }
    
    if verbose:
        print(f"  GGUF-{quant_type}: BPW={bpw:.3f}, MSE={stats['mse']:.6f}")
    
    return stats
