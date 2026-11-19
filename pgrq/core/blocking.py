"""
Stage 2: Block reshaping and norm separation.

Reshapes weight matrices into blocks of size d, extracts L2 norms,
and produces unit direction vectors. The norm is stored exactly (fp16/fp8),
so only the direction is quantized.
"""

import torch
from typing import Tuple, Optional


def reshape_to_blocks(
    W: torch.Tensor,
    block_size: int,
    return_shape_info: bool = True
) -> Tuple[torch.Tensor, Optional[dict]]:
    """
    Reshape a weight matrix into blocks of size block_size.
    
    Args:
        W: Weight tensor of shape (out_features, in_features) or any shape.
        block_size: Block dimension d (must be power of 2).
        return_shape_info: If True, return shape info for reconstruction.
        
    Returns:
        blocks: Tensor of shape (N, block_size) where N = total_elements / block_size
        shape_info: Dict with original shape and padding info (if return_shape_info=True)
    """
    original_shape = W.shape
    W_flat = W.flatten()
    
    total_elements = W_flat.numel()
    
    # Calculate padding needed
    remainder = total_elements % block_size
    if remainder != 0:
        padding_size = block_size - remainder
        W_flat = torch.cat([W_flat, torch.zeros(padding_size, dtype=W.dtype, device=W.device)])
    else:
        padding_size = 0
    
    # Reshape to blocks
    num_blocks = W_flat.numel() // block_size
    blocks = W_flat.view(num_blocks, block_size)
    
    if return_shape_info:
        shape_info = {
            "original_shape": original_shape,
            "total_elements": total_elements,
            "padding_size": padding_size,
            "num_blocks": num_blocks,
            "block_size": block_size,
        }
        return blocks, shape_info
    
    return blocks, None


def reshape_from_blocks(
    blocks: torch.Tensor,
    shape_info: dict,
) -> torch.Tensor:
    """
    Reshape blocks back to original weight matrix shape.
    
    Args:
        blocks: Tensor of shape (N, block_size).
        shape_info: Dict with original shape and padding info.
        
    Returns:
        W: Reconstructed weight tensor of original shape.
    """
    # Flatten blocks
    W_flat = blocks.flatten()
    
    # Remove padding
    if shape_info["padding_size"] > 0:
        W_flat = W_flat[:shape_info["total_elements"]]
    
    # Reshape to original
    W = W_flat.view(shape_info["original_shape"])
    
    return W


def extract_norms(
    blocks: torch.Tensor,
    eps: float = 1e-10,
    norm_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract L2 norms from blocks and produce unit direction vectors.
    
    Stage 2 of PGR-Q: s_b = ||w_b||_2 ; u_b = w_b / (s_b + eps)
    
    Args:
        blocks: Tensor of shape (N, block_size).
        eps: Small constant for numerical stability.
        norm_dtype: Data type for storing norms (float16 or float32).
        
    Returns:
        unit_directions: Tensor of shape (N, block_size), unit vectors.
        norms: Tensor of shape (N, 1), L2 norms stored in norm_dtype.
    """
    # Compute L2 norms
    norms = torch.norm(blocks, p=2, dim=1, keepdim=True)  # (N, 1)
    
    # Normalize to unit vectors
    unit_directions = blocks / (norms + eps)  # (N, block_size)
    
    # Store norms in specified dtype
    norms = norms.to(norm_dtype)
    
    return unit_directions, norms


def apply_norms(
    unit_directions: torch.Tensor,
    norms: torch.Tensor,
) -> torch.Tensor:
    """
    Rescale unit direction vectors by their norms.
    
    Decoder operation: w_b = u_b * s_b
    
    Args:
        unit_directions: Tensor of shape (N, block_size), unit vectors.
        norms: Tensor of shape (N, 1), L2 norms.
        
    Returns:
        blocks: Tensor of shape (N, block_size), rescaled blocks.
    """
    # Ensure norms are in the same dtype as unit_directions for computation
    norms_float = norms.to(unit_directions.dtype)
    
    return unit_directions * norms_float


def split_into_groups(
    Z: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Stage 4: Split rotated blocks into groups.
    
    Zg[n,g,:] = Z[n, g*gd:(g+1)*gd]
    
    Args:
        Z: Tensor of shape (N, block_size), rotated unit directions.
        num_groups: Number of groups G per block.
        
    Returns:
        Zg: Tensor of shape (N, G, gd) where gd = block_size / G.
    """
    N, block_size = Z.shape
    group_dim = block_size // num_groups
    
    # Reshape: (N, block_size) -> (N, G, gd)
    Zg = Z.view(N, num_groups, group_dim)
    
    return Zg


def merge_groups(
    Zg: torch.Tensor,
) -> torch.Tensor:
    """
    Merge groups back into blocks.
    
    Inverse of split_into_groups.
    
    Args:
        Zg: Tensor of shape (N, G, gd).
        
    Returns:
        Z: Tensor of shape (N, block_size).
    """
    N, G, gd = Zg.shape
    return Zg.view(N, G * gd)
