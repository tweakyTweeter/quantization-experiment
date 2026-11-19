"""
GPTQ implementation (Frantar et al., 2022).

Core algorithm: layer-wise quantization using inverse Hessian information
to find optimal rounding decisions. This is a faithful CPU implementation
without requiring CUDA kernels.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple


class GPTQ:
    """
    GPTQ quantizer for a single linear layer.
    
    Implements the OBQ/GPTQ algorithm:
    1. Collect Hessian H = X^T X from calibration data
    2. For each column (in order), find optimal quantized value
       considering the Hessian-weighted error propagation
    3. Update remaining weights to compensate for quantization error
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        percdamp: float = 0.01,
        block_size: int = 128,
    ):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.percdamp = percdamp
        self.block_size = block_size
        
        self.rows = layer.weight.shape[0]
        self.cols = layer.weight.shape[1]
        
        # Hessian accumulator
        self.H = torch.zeros((self.cols, self.cols), dtype=torch.float32)
        self.nsamples = 0
    
    def add_batch(self, inp: torch.Tensor):
        """
        Add a batch of inputs to the Hessian estimate.
        inp: (batch, seq_len, in_features) or (batch, in_features)
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        
        inp = inp.float()
        n = inp.shape[0]
        
        self.H += inp.T @ inp
        self.nsamples += n
    
    def quantize(self) -> Tuple[torch.Tensor, Dict]:
        """
        Run GPTQ quantization.
        
        Returns:
            quantized_weight: Dequantized weight tensor
            info: Dict with quantization metadata
        """
        W = self.layer.weight.data.float().clone()
        
        # Normalize Hessian
        H = self.H / self.nsamples
        
        # Dampening
        damp = self.percdamp * torch.diag(H).mean()
        H += damp * torch.eye(self.cols, dtype=H.dtype)
        
        # Cholesky decomposition of H inverse
        try:
            Hinv = torch.linalg.cholesky(torch.linalg.inv(H))
        except:
            # Fallback: add more dampening
            H += 0.1 * torch.eye(self.cols, dtype=H.dtype)
            try:
                Hinv = torch.linalg.cholesky(torch.linalg.inv(H))
            except:
                # Last resort: just use diagonal
                Hinv = torch.diag(1.0 / torch.sqrt(torch.diag(H) + 1e-8))
        
        Hinv = Hinv.T  # Upper triangular
        
        # Quantize column by column in blocks
        Q = torch.zeros_like(W)
        Losses = torch.zeros(self.rows, dtype=torch.float32)
        
        for i1 in range(0, self.cols, self.block_size):
            i2 = min(i1 + self.block_size, self.cols)
            count = i2 - i1
            
            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            Hinv_block = Hinv[i1:i2, i1:i2]
            
            for j in range(count):
                w = W_block[:, j]
                d = Hinv_block[j, j]
                
                # Determine quantization grid for this column
                col_idx = i1 + j
                group_idx = col_idx // self.group_size
                
                # Compute scale and zero point for this group
                if j % self.group_size == 0 or j == 0:
                    # New group: compute scale
                    g_start = j
                    g_end = min(j + self.group_size, count)
                    group_w = W_block[:, g_start:g_end]
                    
                    if self.sym:
                        maxval = group_w.abs().max(dim=1).values.clamp(min=1e-8)
                        scale = maxval / (2 ** (self.bits - 1) - 1)
                        zero = torch.zeros_like(scale)
                    else:
                        minval = group_w.min(dim=1).values
                        maxval = group_w.max(dim=1).values
                        scale = (maxval - minval).clamp(min=1e-8) / (2**self.bits - 1)
                        zero = torch.round(-minval / scale)
                
                # Quantize
                if self.sym:
                    q = torch.clamp(
                        torch.round(w / scale),
                        -(2**(self.bits-1) - 1),
                        2**(self.bits-1) - 1
                    )
                    q_val = q * scale
                else:
                    q = torch.clamp(
                        torch.round(w / scale + zero),
                        0, 2**self.bits - 1
                    )
                    q_val = (q - zero) * scale
                
                Q_block[:, j] = q_val
                Err_block[:, j] = (w - q_val) / d
                
                # Update remaining columns in this block
                if j + 1 < count:
                    W_block[:, j+1:] -= Err_block[:, j:j+1] @ Hinv_block[j:j+1, j+1:]
            
            Q[:, i1:i2] = Q_block
            
            # Propagate error to remaining columns
            if i2 < self.cols:
                W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]
        
        # Compute total quantization error
        error = ((self.layer.weight.data.float() - Q) ** 2).mean().item()
        
        return Q, {
            'bits': self.bits,
            'group_size': self.group_size,
            'mse': error,
            'sym': self.sym,
        }


def quantize_model_gptq(
    model,
    tokenizer,
    bits: int = 4,
    group_size: int = 128,
    calibration_texts: Optional[List[str]] = None,
    n_samples: int = 128,
    seq_len: int = 2048,
    verbose: bool = True,
) -> Dict:
    """
    Quantize a HuggingFace model using GPTQ.
    
    Args:
        model: HuggingFace causal LM model
        tokenizer: Tokenizer
        bits: Target bits per weight
        group_size: Quantization group size
        calibration_texts: Calibration text samples
        n_samples: Number of calibration samples
        seq_len: Sequence length for calibration
        verbose: Print progress
        
    Returns:
        stats: Quantization statistics
    """
    from datasets import load_dataset
    from tqdm import tqdm
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get calibration data
    if calibration_texts is None:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in dataset["text"] if len(t.strip()) > 100]
        calibration_texts = texts[:n_samples]
    
    # Tokenize calibration data
    encodings = tokenizer(
        calibration_texts[:n_samples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    )
    cal_input_ids = encodings.input_ids.to(device)
    
    # Collect layer inputs via hooks
    # We'll quantize layer by layer
    layers_to_quantize = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'embed' not in name.lower() and 'lm_head' not in name.lower() and 'norm' not in name.lower():
                layers_to_quantize.append((name, module))
    
    if verbose:
        print(f"GPTQ: Quantizing {len(layers_to_quantize)} layers at {bits}-bit (group_size={group_size})")
    
    # For each layer, collect inputs and run GPTQ
    total_mse = 0.0
    total_params = 0
    
    # Simple approach: collect all activations in a forward pass, then quantize
    layer_inputs = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in layer_inputs:
                layer_inputs[name] = []
            # Store input activations
            inp = input[0].detach().float()
            layer_inputs[name].append(inp)
        return hook_fn
    
    for name, module in layers_to_quantize:
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)
    
    # Forward pass to collect activations
    with torch.no_grad():
        # Process in small batches to save memory
        batch_size = 4
        for i in range(0, min(n_samples, cal_input_ids.shape[0]), batch_size):
            batch = cal_input_ids[i:i+batch_size]
            if batch.shape[1] > seq_len:
                batch = batch[:, :seq_len]
            try:
                model(batch)
            except:
                pass  # Some batches may fail due to padding
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Now quantize each layer
    for name, module in tqdm(layers_to_quantize, desc="GPTQ", disable=not verbose):
        if name not in layer_inputs or len(layer_inputs[name]) == 0:
            continue
        
        gptq = GPTQ(module, bits=bits, group_size=group_size)
        
        # Add all collected inputs
        for inp in layer_inputs[name]:
            gptq.add_batch(inp)
        
        # Free memory
        del layer_inputs[name]
        
        # Quantize
        Q, info = gptq.quantize()
        
        # Replace weight
        module.weight.data = Q.to(module.weight.dtype)
        
        total_mse += info['mse'] * module.weight.numel()
        total_params += module.weight.numel()
    
    # Compute effective BPW
    # GPTQ stores: quantized weights (bits per element) + scales (fp16 per group) + zeros (if asymmetric)
    total_weight_bits = total_params * bits
    num_groups = total_params / group_size
    scale_bits = num_groups * 16  # fp16 scales
    zero_bits = 0 if True else num_groups * bits  # sym = no zeros
    total_bits = total_weight_bits + scale_bits + zero_bits
    bpw = total_bits / total_params
    
    stats = {
        'method': f'GPTQ-{bits}b',
        'bits': bits,
        'group_size': group_size,
        'bpw': bpw,
        'mse': total_mse / total_params if total_params > 0 else 0,
        'total_params': total_params,
    }
    
    if verbose:
        print(f"  GPTQ-{bits}b: BPW={bpw:.3f}, MSE={stats['mse']:.6f}")
    
    return stats
