"""
Stage 3: Fast Walsh-Hadamard Transform (FWHT) rotation.

Implements random-signed Hadamard rotation for incoherence:
- D = random sign diagonal (deterministic from seed)
- H = normalized Hadamard matrix (applied via butterfly FWHT)

The FWHT is O(d log d) and implemented via iterative butterfly operations,
NOT dense matrix multiplication.
"""

import torch
from typing import Optional, Tuple
import math


def _generate_signs(size: int, seed: int, device: torch.device) -> torch.Tensor:
    """
    Generate deterministic random signs {-1, +1} from seed.
    
    Args:
        size: Number of signs to generate.
        seed: Random seed for reproducibility.
        device: Target device.
        
    Returns:
        signs: Tensor of shape (size,) with values in {-1, +1}.
    """
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    
    # Generate random bits and convert to signs
    bits = torch.randint(0, 2, (size,), generator=generator, device='cpu')
    signs = 2 * bits - 1  # Convert {0, 1} to {-1, +1}
    
    return signs.float().to(device)


def fwht(X: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform using iterative butterfly operations.
    
    This is O(d log d) and works correctly for batched inputs.
    
    The transform computes H @ X where H is the Hadamard matrix.
    With normalize=True, computes (1/sqrt(d)) * H @ X.
    
    Args:
        X: Input tensor of shape (..., d) where d is power of 2.
        normalize: If True, normalize by 1/sqrt(d).
        
    Returns:
        Y: Transformed tensor of same shape as X.
    """
    *batch_dims, d = X.shape
    
    # Verify d is power of 2
    if d & (d - 1) != 0:
        raise ValueError(f"Last dimension must be power of 2, got {d}")
    
    # Work with a copy to avoid modifying input
    Y = X.clone()
    
    # Reshape for batch processing: (batch, d)
    original_shape = Y.shape
    Y = Y.reshape(-1, d)
    batch_size = Y.shape[0]
    
    # Standard iterative FWHT butterfly
    # At each stage, we combine pairs of elements
    h = 1
    while h < d:
        # Process all pairs at distance h
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                # Butterfly operation for all batch elements
                a = Y[:, j].clone()
                b = Y[:, j + h].clone()
                Y[:, j] = a + b
                Y[:, j + h] = a - b
        h *= 2
    
    # Normalize
    if normalize:
        Y = Y / math.sqrt(d)
    
    # Restore original shape
    Y = Y.reshape(original_shape)
    
    return Y


def fwht_inverse(Y: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Inverse Fast Walsh-Hadamard Transform.
    
    The Hadamard matrix is self-inverse (H @ H = d * I), so:
    - Forward: Y = (1/sqrt(d)) * H @ X
    - Inverse: X = (1/sqrt(d)) * H @ Y
    
    With our normalization, FWHT is its own inverse.
    
    Args:
        Y: Input tensor of shape (..., d) where d is power of 2.
        normalize: If True, normalize by 1/sqrt(d).
        
    Returns:
        X: Inverse transformed tensor of same shape as Y.
    """
    # For normalized Hadamard, forward and inverse are the same
    # (H/sqrt(d)) @ (H/sqrt(d)) = H @ H / d = d * I / d = I
    return fwht(Y, normalize=normalize)


def fwht_rotate(
    U: torch.Tensor,
    seed: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Apply random-signed Hadamard rotation for incoherence (Stage 3).
    
    Z = (1/sqrt(d)) * H @ D @ U
    
    where D is a diagonal matrix of random signs determined by seed.
    
    Args:
        U: Unit direction vectors of shape (N, d) or (..., d).
        seed: Random seed for sign generation.
        normalize: If True, normalize by 1/sqrt(d).
        
    Returns:
        Z: Rotated vectors of same shape as U.
    """
    *batch_dims, d = U.shape
    
    # Generate random signs
    signs = _generate_signs(d, seed, U.device)
    
    # Apply sign diagonal: X = D @ U (elementwise multiplication)
    X = U * signs
    
    # Apply FWHT
    Z = fwht(X, normalize=normalize)
    
    return Z


def fwht_rotate_inverse(
    Z: torch.Tensor,
    seed: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Inverse of random-signed Hadamard rotation.
    
    U = D^{-1} @ H @ Z * sqrt(d) (when normalized forward)
    
    Since D is diagonal with +/-1, D^{-1} = D.
    Since H is symmetric and H @ H = d * I:
    - If forward was Z = (1/sqrt(d)) * H @ D @ U
    - Then U = D @ (1/sqrt(d)) * H @ Z (applying H then D)
    
    Args:
        Z: Rotated vectors of shape (N, d) or (..., d).
        seed: Random seed for sign generation (must match forward).
        normalize: If True, use normalized inverse.
        
    Returns:
        U: Original unit direction vectors of same shape as Z.
    """
    *batch_dims, d = Z.shape
    
    # Generate random signs (same as forward)
    signs = _generate_signs(d, seed, Z.device)
    
    # Apply inverse FWHT first
    X = fwht_inverse(Z, normalize=normalize)
    
    # Apply sign diagonal (D^{-1} = D since signs are +/-1)
    U = X * signs
    
    return U


# Alias for clarity
fwht_inverse_rotate = fwht_rotate_inverse


def verify_fwht_roundtrip(
    U: torch.Tensor,
    seed: int,
    tol: float = 1e-10,
) -> Tuple[bool, float]:
    """
    Verify that FWHT rotation round-trip error is below tolerance.
    
    Required verification: error < 1e-10.
    
    Args:
        U: Input tensor.
        seed: Random seed.
        tol: Tolerance for round-trip error.
        
    Returns:
        passed: True if error < tol.
        max_error: Maximum absolute error.
    """
    Z = fwht_rotate(U, seed)
    U_rec = fwht_rotate_inverse(Z, seed)
    
    max_error = (U - U_rec).abs().max().item()
    passed = max_error < tol
    
    return passed, max_error
