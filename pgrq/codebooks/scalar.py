"""
Stage 7b: Scalar floor quantization.

Two interchangeable schemes:
1. TurboQuant Beta-Max-Lloyd (RECOMMENDED default)
   - Post-rotation coords ~ Beta((d-3)/2 shape) ≈ N(0,1/d)
   - Precompute optimal scalar codebook by 1-D Lloyd-Max

2. Digit-plane / base-p (EXPERIMENTAL)
   - q = round(e/scale + (2^L-1)/2) clipped to [0,2^L-1]
   - Keep L low-order bit-planes

MSE targets for unit vectors (Beta-Max-Lloyd):
- b=1: ~0.36
- b=2: ~0.117  
- b=3: ~0.03
- b=4: ~0.009
"""

import torch
from typing import Tuple, Optional
import math


def _lloyd_max_1d(
    pdf_samples: torch.Tensor,
    num_levels: int,
    num_iters: int = 100,
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd-Max scalar quantizer design for a given distribution.
    
    Minimizes MSE: sum over voronoi regions of integral of (x - c_i)^2 * p(x) dx
    
    We approximate this with samples from the distribution.
    
    Args:
        pdf_samples: Samples from the target distribution.
        num_levels: Number of quantization levels (2^bits).
        num_iters: Maximum iterations.
        tol: Convergence tolerance.
        
    Returns:
        centroids: Optimal centroids (num_levels,).
        boundaries: Decision boundaries (num_levels - 1,).
    """
    # Sort samples
    samples = pdf_samples.sort().values
    N = len(samples)
    
    # Initialize centroids uniformly over sample range
    min_val, max_val = samples.min().item(), samples.max().item()
    centroids = torch.linspace(min_val, max_val, num_levels, device=samples.device)
    
    for _ in range(num_iters):
        # Compute boundaries as midpoints
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        
        # Assign samples to nearest centroid
        # boundaries: (L-1,), samples: (N,)
        # Find which bin each sample falls into
        assignments = torch.bucketize(samples, boundaries)  # (N,), values in [0, L-1]
        
        # Update centroids as conditional means
        new_centroids = torch.zeros_like(centroids)
        for i in range(num_levels):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = samples[mask].mean()
            else:
                new_centroids[i] = centroids[i]
        
        # Check convergence
        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        
        if shift < tol:
            break
    
    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    
    return centroids, boundaries


def _sample_from_beta_distribution(
    d: int,
    num_samples: int,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample from the distribution of post-Hadamard coordinates.
    
    For unit vectors on S^{d-1}, after Hadamard rotation, each coordinate
    follows approximately Beta((d-1)/2, (d-1)/2) rescaled to [-1/sqrt(d), 1/sqrt(d)].
    
    For large d, this approaches N(0, 1/d).
    
    Args:
        d: Block dimension.
        num_samples: Number of samples to generate.
        seed: Random seed.
        device: Target device.
        
    Returns:
        samples: Samples from the coordinate distribution.
    """
    torch.manual_seed(seed)
    
    # Shape parameter for Beta distribution
    alpha = (d - 1) / 2
    
    if alpha < 0.5:
        alpha = 0.5  # Minimum valid shape
    
    # Sample from Beta(alpha, alpha) which is symmetric around 0.5
    # Then rescale to [-1, 1] and divide by sqrt(d)
    
    # Use torch.distributions for Beta sampling
    beta_dist = torch.distributions.Beta(alpha, alpha)
    beta_samples = beta_dist.sample((num_samples,))
    
    # Rescale from [0, 1] to [-1, 1]
    samples = 2 * beta_samples - 1
    
    # Scale by 1/sqrt(d) to match post-Hadamard variance
    samples = samples / math.sqrt(d)
    
    return samples.to(device)


class BetaMaxLloydQuantizer:
    """
    Beta-Max-Lloyd scalar quantizer (Stage 7b option i).
    
    Optimal scalar quantizer for post-Hadamard coordinates.
    Precomputes Lloyd-Max codebook for the Beta distribution.
    """
    
    # Precomputed optimal MSE per unit vector for sanity checking
    EXPECTED_MSE = {
        1: 0.36,   # Rough targets
        2: 0.117,
        3: 0.03,
        4: 0.009,
    }
    
    def __init__(
        self,
        bits: int,
        block_size: int = 64,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        Initialize Beta-Max-Lloyd quantizer.
        
        Args:
            bits: Bits per coordinate (1, 2, 3, or 4).
            block_size: Block dimension d (for distribution shape).
            device: Target device.
            seed: Random seed for codebook design.
        """
        self.bits = bits
        self.num_levels = 2 ** bits
        self.block_size = block_size
        self.device = device
        
        # Design optimal codebook
        self._design_codebook(seed)
    
    def _design_codebook(self, seed: int) -> None:
        """Design optimal Lloyd-Max codebook."""
        # Generate samples from post-Hadamard distribution
        num_samples = 100000
        samples = _sample_from_beta_distribution(
            self.block_size, num_samples, seed=seed, device=self.device
        )
        
        # Run Lloyd-Max algorithm
        self.centroids, self.boundaries = _lloyd_max_1d(
            samples, self.num_levels, num_iters=200
        )
        
        # Ensure centroids are sorted
        self.centroids = self.centroids.sort().values
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2
    
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode values to nearest centroid indices.
        
        Args:
            X: Values of shape (...,).
            
        Returns:
            codes: Index tensor of shape (...,) with values in [0, 2^bits).
        """
        boundaries = self.boundaries.to(X.device)
        codes = torch.bucketize(X, boundaries)
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode centroid indices back to values.
        
        Args:
            codes: Index tensor of shape (...,) with values in [0, 2^bits).
            
        Returns:
            X: Reconstructed values of shape (...,).
        """
        centroids = self.centroids.to(codes.device)
        return centroids[codes]
    
    def quantize(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize values (encode + decode).
        
        Args:
            X: Values of shape (...,).
            
        Returns:
            codes: Index tensor.
            reconstructed: Reconstructed values.
        """
        codes = self.encode(X)
        reconstructed = self.decode(codes)
        return codes, reconstructed
    
    def get_mse(self, X: torch.Tensor) -> float:
        """Compute MSE on data."""
        _, reconstructed = self.quantize(X)
        return ((X - reconstructed) ** 2).mean().item()
    
    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "bits": self.bits,
            "block_size": self.block_size,
            "centroids": self.centroids.cpu(),
            "boundaries": self.boundaries.cpu(),
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.bits = state["bits"]
        self.block_size = state["block_size"]
        self.num_levels = 2 ** self.bits
        self.centroids = state["centroids"].to(self.device)
        self.boundaries = state["boundaries"].to(self.device)
    
    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> "BetaMaxLloydQuantizer":
        """Create quantizer from state dict without recomputing codebook."""
        q = cls.__new__(cls)
        q.device = device
        q.load_state_dict(state)
        return q


class DigitPlaneQuantizer:
    """
    Digit-plane / base-p scalar quantizer (Stage 7b option ii).
    
    EXPERIMENTAL alternative to Beta-Max-Lloyd.
    
    q = round(e/scale + (2^L-1)/2) clipped to [0, 2^L-1]
    recon = (q - (2^L-1)/2) * scale
    """
    
    def __init__(
        self,
        levels: int = 4,
        scale: Optional[float] = None,
        block_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize digit-plane quantizer.
        
        Args:
            levels: L, number of bit-planes (bits per coordinate).
            scale: Quantization scale. If None, auto-computed.
            block_size: Block dimension d.
            device: Target device.
        """
        self.levels = levels
        self.num_levels = 2 ** levels
        self.block_size = block_size
        self.device = device
        
        # Auto-compute scale based on expected range of post-Hadamard coords
        if scale is None:
            # Expected range is roughly [-3/sqrt(d), 3/sqrt(d)] for 99.7% of values
            max_val = 3.0 / math.sqrt(block_size)
            self.scale = 2 * max_val / (self.num_levels - 1)
        else:
            self.scale = scale
        
        self.offset = (self.num_levels - 1) / 2
    
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode values to digit-plane codes.
        
        q = round(x/scale + offset) clipped to [0, 2^L - 1]
        
        Args:
            X: Values of shape (...,).
            
        Returns:
            codes: Index tensor of shape (...,) with values in [0, 2^L).
        """
        q = torch.round(X / self.scale + self.offset)
        q = q.clamp(0, self.num_levels - 1).long()
        return q
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode digit-plane codes back to values.
        
        x = (q - offset) * scale
        
        Args:
            codes: Index tensor of shape (...,) with values in [0, 2^L).
            
        Returns:
            X: Reconstructed values of shape (...,).
        """
        return (codes.float() - self.offset) * self.scale
    
    def quantize(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize values (encode + decode).
        
        Args:
            X: Values of shape (...,).
            
        Returns:
            codes: Index tensor.
            reconstructed: Reconstructed values.
        """
        codes = self.encode(X)
        reconstructed = self.decode(codes)
        return codes, reconstructed
    
    def get_mse(self, X: torch.Tensor) -> float:
        """Compute MSE on data."""
        _, reconstructed = self.quantize(X)
        return ((X - reconstructed) ** 2).mean().item()
    
    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "levels": self.levels,
            "scale": self.scale,
            "block_size": self.block_size,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.levels = state["levels"]
        self.num_levels = 2 ** self.levels
        self.scale = state["scale"]
        self.block_size = state["block_size"]
        self.offset = (self.num_levels - 1) / 2
    
    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> "DigitPlaneQuantizer":
        """Create quantizer from state dict."""
        q = cls(
            levels=state["levels"],
            scale=state["scale"],
            block_size=state["block_size"],
            device=device,
        )
        return q


def quantize_scalar_floor(
    residual: torch.Tensor,
    method: str = "beta_max_lloyd",
    bits: int = 2,
    block_size: int = 64,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, object]:
    """
    Apply scalar floor quantization to residuals.
    
    Args:
        residual: Residual tensor of shape (..., D).
        method: "beta_max_lloyd" or "digit_planes".
        bits: Bits per coordinate.
        block_size: Block dimension.
        device: Target device.
        
    Returns:
        codes: Scalar codes of shape (..., D).
        reconstructed: Reconstructed residual.
        quantizer: The quantizer object for serialization.
    """
    if method == "beta_max_lloyd":
        quantizer = BetaMaxLloydQuantizer(bits=bits, block_size=block_size, device=device)
    elif method == "digit_planes":
        quantizer = DigitPlaneQuantizer(levels=bits, block_size=block_size, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Flatten, quantize, reshape
    original_shape = residual.shape
    flat = residual.flatten()
    
    codes, reconstructed = quantizer.quantize(flat)
    
    codes = codes.view(original_shape)
    reconstructed = reconstructed.view(original_shape)
    
    return codes, reconstructed, quantizer
