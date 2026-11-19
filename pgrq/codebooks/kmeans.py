"""
Stage 5: Base codebook using K-means clustering.

Implements learned codebooks with nearest-centroid encoding.
NO hash-routing - always argmin over all centroids.
"""

import torch
from typing import Optional, Tuple
import math


def _kmeans_plusplus_init(
    X: torch.Tensor,
    K: int,
    seed: int,
) -> torch.Tensor:
    """
    K-means++ initialization for better centroid starting points.
    
    Args:
        X: Data tensor of shape (N, D).
        K: Number of centroids.
        seed: Random seed.
        
    Returns:
        centroids: Initial centroids of shape (K, D).
    """
    N, D = X.shape
    device = X.device
    
    generator = torch.Generator(device=device if device.type != 'mps' else 'cpu')
    generator.manual_seed(seed)
    
    # Use CPU generator for indexing
    cpu_gen = torch.Generator(device='cpu')
    cpu_gen.manual_seed(seed)
    
    centroids = torch.zeros(K, D, device=device, dtype=X.dtype)
    
    # Choose first centroid uniformly at random
    idx = torch.randint(0, N, (1,), generator=cpu_gen).item()
    centroids[0] = X[idx]
    
    # Choose remaining centroids
    for k in range(1, K):
        # Compute distances to nearest centroid
        # Shape: (N, k) -> (N,)
        dists = torch.cdist(X, centroids[:k])  # (N, k)
        min_dists = dists.min(dim=1).values  # (N,)
        min_dists_sq = min_dists ** 2
        
        # Sample proportional to squared distance
        probs = min_dists_sq / (min_dists_sq.sum() + 1e-10)
        probs_cpu = probs.cpu()
        idx = torch.multinomial(probs_cpu, 1, generator=cpu_gen).item()
        centroids[k] = X[idx]
    
    return centroids


def _random_init(
    X: torch.Tensor,
    K: int,
    seed: int,
) -> torch.Tensor:
    """
    Random initialization: sample K points from X.
    """
    N = X.shape[0]
    
    cpu_gen = torch.Generator(device='cpu')
    cpu_gen.manual_seed(seed)
    
    indices = torch.randperm(N, generator=cpu_gen)[:K]
    return X[indices].clone()


def train_kmeans(
    X: torch.Tensor,
    K: int,
    num_iters: int = 100,
    init: str = "kmeans++",
    seed: int = 42,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train K-means codebook.
    
    Args:
        X: Data tensor of shape (N, D).
        K: Number of centroids.
        num_iters: Maximum number of iterations.
        init: Initialization method ("kmeans++" or "random").
        seed: Random seed.
        tol: Convergence tolerance (relative change in centroids).
        verbose: Print progress.
        
    Returns:
        centroids: Learned centroids of shape (K, D).
        assignments: Final assignments of shape (N,).
    """
    N, D = X.shape
    device = X.device
    
    # Handle edge cases
    if N <= K:
        # If we have fewer points than centroids, just use the points
        centroids = torch.zeros(K, D, device=device, dtype=X.dtype)
        centroids[:N] = X
        # Fill remaining with small random values
        if N < K:
            centroids[N:] = X.mean(dim=0) + torch.randn(K - N, D, device=device) * 0.01
        assignments = torch.arange(N, device=device)
        return centroids, assignments
    
    # Initialize centroids
    if init == "kmeans++":
        centroids = _kmeans_plusplus_init(X, K, seed)
    else:
        centroids = _random_init(X, K, seed)
    
    prev_centroids = centroids.clone()
    
    for iteration in range(num_iters):
        # Assignment step: find nearest centroid for each point
        # Use batched cdist for memory efficiency on large datasets
        batch_size = min(10000, N)
        assignments = torch.zeros(N, device=device, dtype=torch.long)
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            dists = torch.cdist(X[i:end], centroids)  # (batch, K)
            assignments[i:end] = dists.argmin(dim=1)
        
        # Update step: recompute centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device)
        
        for k in range(K):
            mask = assignments == k
            count = mask.sum()
            counts[k] = count
            if count > 0:
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # Empty cluster: reinitialize from a random point
                cpu_gen = torch.Generator(device='cpu')
                cpu_gen.manual_seed(seed + iteration * K + k)
                idx = torch.randint(0, N, (1,), generator=cpu_gen).item()
                new_centroids[k] = X[idx]
        
        # Check convergence
        centroid_shift = (new_centroids - prev_centroids).norm(dim=1).max().item()
        centroid_norm = prev_centroids.norm(dim=1).max().item() + 1e-10
        relative_shift = centroid_shift / centroid_norm
        
        centroids = new_centroids
        prev_centroids = centroids.clone()
        
        if verbose and (iteration + 1) % 10 == 0:
            mse = compute_mse(X, centroids, assignments)
            print(f"K-means iter {iteration + 1}: MSE={mse:.6f}, shift={relative_shift:.2e}")
        
        if relative_shift < tol:
            if verbose:
                print(f"K-means converged at iteration {iteration + 1}")
            break
    
    return centroids, assignments


def compute_mse(
    X: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
) -> float:
    """Compute mean squared error of quantization."""
    reconstructed = centroids[assignments]
    mse = ((X - reconstructed) ** 2).mean().item()
    return mse


class KMeansCodebook:
    """
    K-means codebook for base quantization (Stage 5).
    
    Encoding is always nearest-centroid (argmin), never hash-routed.
    """
    
    def __init__(
        self,
        K: int,
        D: int,
        device: str = "cpu",
    ):
        """
        Initialize codebook.
        
        Args:
            K: Number of centroids.
            D: Dimension of each centroid.
            device: Target device.
        """
        self.K = K
        self.D = D
        self.device = device
        self.centroids: Optional[torch.Tensor] = None
        self._trained = False
    
    def train(
        self,
        X: torch.Tensor,
        num_iters: int = 100,
        init: str = "kmeans++",
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """
        Train the codebook on data.
        
        Args:
            X: Training data of shape (N, D).
            num_iters: Maximum k-means iterations.
            init: Initialization method.
            seed: Random seed.
            verbose: Print progress.
        """
        if X.shape[1] != self.D:
            raise ValueError(f"Expected D={self.D}, got {X.shape[1]}")
        
        X = X.to(self.device)
        
        self.centroids, _ = train_kmeans(
            X, self.K, num_iters=num_iters, init=init, seed=seed, verbose=verbose
        )
        self._trained = True
    
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode data points to nearest centroid indices.
        
        Args:
            X: Data tensor of shape (..., D).
            
        Returns:
            codes: Index tensor of shape (...,) with values in [0, K).
        """
        if not self._trained:
            raise RuntimeError("Codebook not trained")
        
        original_shape = X.shape[:-1]
        X_flat = X.view(-1, self.D)
        
        # Batched distance computation
        N = X_flat.shape[0]
        batch_size = min(10000, N)
        codes = torch.zeros(N, device=X.device, dtype=torch.long)
        
        centroids = self.centroids.to(X.device)
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            dists = torch.cdist(X_flat[i:end], centroids)
            codes[i:end] = dists.argmin(dim=1)
        
        return codes.view(original_shape)
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode centroid indices back to vectors.
        
        Args:
            codes: Index tensor of shape (...,) with values in [0, K).
            
        Returns:
            X: Reconstructed tensor of shape (..., D).
        """
        if not self._trained:
            raise RuntimeError("Codebook not trained")
        
        centroids = self.centroids.to(codes.device)
        return centroids[codes]
    
    def quantize(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize data (encode + decode).
        
        Args:
            X: Data tensor of shape (..., D).
            
        Returns:
            codes: Index tensor.
            reconstructed: Reconstructed tensor.
        """
        codes = self.encode(X)
        reconstructed = self.decode(codes)
        return codes, reconstructed
    
    def get_mse(self, X: torch.Tensor) -> float:
        """Compute MSE on data."""
        codes = self.encode(X)
        reconstructed = self.decode(codes)
        return ((X - reconstructed) ** 2).mean().item()
    
    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "K": self.K,
            "D": self.D,
            "centroids": self.centroids.cpu() if self.centroids is not None else None,
            "trained": self._trained,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.K = state["K"]
        self.D = state["D"]
        self.centroids = state["centroids"].to(self.device) if state["centroids"] is not None else None
        self._trained = state["trained"]
    
    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> "KMeansCodebook":
        """Create codebook from state dict."""
        cb = cls(K=state["K"], D=state["D"], device=device)
        cb.load_state_dict(state)
        return cb
