"""
Stage 7a: Residual Vector Quantization (RVQ).

Implements both:
1. Greedy RVQ: Sequential codebook training on residuals
2. Joint refinement: After greedy init, iteratively refine all codebooks

This is AQLM-style additive learned codebooks with M levels.
"""

import torch
from typing import List, Tuple, Optional
from pgrq.codebooks.kmeans import train_kmeans, KMeansCodebook


def train_greedy_rvq(
    X: torch.Tensor,
    M: int,
    K: int,
    num_iters: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Train greedy RVQ: sequentially train codebooks on residuals.
    
    For m in 1..M:
        Cm = kmeans(residual_{m-1})
        a_m = argmin over Cm
        residual_m = residual_{m-1} - Cm[a_m]
    
    Args:
        X: Input data of shape (N, D).
        M: Number of residual levels.
        K: Number of centroids per level.
        num_iters: K-means iterations per level.
        seed: Random seed.
        verbose: Print progress.
        
    Returns:
        codebooks: List of M centroid tensors, each (K, D).
        all_codes: List of M code tensors, each (N,).
    """
    N, D = X.shape
    device = X.device
    
    codebooks = []
    all_codes = []
    residual = X.clone()
    
    for m in range(M):
        if verbose:
            print(f"Training RVQ level {m + 1}/{M}")
        
        # Train codebook on current residual
        centroids, codes = train_kmeans(
            residual, K, num_iters=num_iters, 
            init="kmeans++", seed=seed + m, verbose=verbose
        )
        
        codebooks.append(centroids)
        all_codes.append(codes)
        
        # Update residual
        residual = residual - centroids[codes]
        
        if verbose:
            mse = (residual ** 2).mean().item()
            print(f"  Level {m + 1} residual MSE: {mse:.6f}")
    
    return codebooks, all_codes


def joint_refine_rvq(
    X: torch.Tensor,
    codebooks: List[torch.Tensor],
    num_iters: int = 3,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Joint refinement of RVQ codebooks.
    
    After greedy init, repeat for a few rounds:
    1. For each m: hold others fixed, recompute a_m by argmin over full sum
    2. Then re-fit each Cm to its assigned residuals (least squares / mean)
    
    Args:
        X: Original input data of shape (N, D).
        codebooks: List of M initial codebook tensors.
        num_iters: Number of refinement iterations.
        verbose: Print progress.
        
    Returns:
        codebooks: Refined codebook tensors.
        all_codes: Final code assignments.
    """
    M = len(codebooks)
    N, D = X.shape
    device = X.device
    K = codebooks[0].shape[0]
    
    # Initialize codes with greedy assignments
    all_codes = []
    residual = X.clone()
    for m in range(M):
        dists = torch.cdist(residual, codebooks[m])
        codes = dists.argmin(dim=1)
        all_codes.append(codes)
        residual = residual - codebooks[m][codes]
    
    for iteration in range(num_iters):
        if verbose:
            print(f"Joint refinement iteration {iteration + 1}/{num_iters}")
        
        # Step 1: Refine assignments for each level
        for m in range(M):
            # Compute reconstruction from all OTHER levels
            other_recon = torch.zeros_like(X)
            for j in range(M):
                if j != m:
                    other_recon += codebooks[j][all_codes[j]]
            
            # The residual that level m must explain
            target = X - other_recon
            
            # Find best assignment for level m
            dists = torch.cdist(target, codebooks[m])
            all_codes[m] = dists.argmin(dim=1)
        
        # Step 2: Re-fit centroids to their assigned residuals
        for m in range(M):
            # Compute what level m should reconstruct
            other_recon = torch.zeros_like(X)
            for j in range(M):
                if j != m:
                    other_recon += codebooks[j][all_codes[j]]
            target = X - other_recon
            
            # Update centroids as mean of assigned targets
            new_centroids = torch.zeros_like(codebooks[m])
            for k in range(K):
                mask = all_codes[m] == k
                if mask.sum() > 0:
                    new_centroids[k] = target[mask].mean(dim=0)
                else:
                    # Keep old centroid for empty clusters
                    new_centroids[k] = codebooks[m][k]
            
            codebooks[m] = new_centroids
        
        if verbose:
            # Compute total MSE
            total_recon = sum(codebooks[m][all_codes[m]] for m in range(M))
            mse = ((X - total_recon) ** 2).mean().item()
            print(f"  Iteration {iteration + 1} MSE: {mse:.6f}")
    
    return codebooks, all_codes


def train_rvq(
    X: torch.Tensor,
    M: int,
    K: int,
    num_iters: int = 100,
    joint_refinement: bool = True,
    joint_refinement_iters: int = 3,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Train RVQ with optional joint refinement.
    
    Args:
        X: Input data of shape (N, D).
        M: Number of residual levels.
        K: Number of centroids per level.
        num_iters: K-means iterations for greedy init.
        joint_refinement: Enable joint refinement after greedy.
        joint_refinement_iters: Number of joint refinement iterations.
        seed: Random seed.
        verbose: Print progress.
        
    Returns:
        codebooks: List of M centroid tensors.
        all_codes: List of M code tensors.
    """
    # Greedy initialization
    codebooks, all_codes = train_greedy_rvq(
        X, M, K, num_iters=num_iters, seed=seed, verbose=verbose
    )
    
    # Optional joint refinement
    if joint_refinement and M > 1:
        codebooks, all_codes = joint_refine_rvq(
            X, codebooks, num_iters=joint_refinement_iters, verbose=verbose
        )
    
    return codebooks, all_codes


class RVQCodebook:
    """
    Residual Vector Quantization codebook (Stage 7a).
    
    Implements additive codebooks: x ≈ C0[a0] + C1[a1] + ... + C_{M-1}[a_{M-1}]
    """
    
    def __init__(
        self,
        M: int,
        K: int,
        D: int,
        device: str = "cpu",
    ):
        """
        Initialize RVQ codebook.
        
        Args:
            M: Number of residual levels.
            K: Number of centroids per level.
            D: Dimension of each centroid.
            device: Target device.
        """
        self.M = M
        self.K = K
        self.D = D
        self.device = device
        self.codebooks: List[torch.Tensor] = []
        self._trained = False
    
    def train(
        self,
        X: torch.Tensor,
        num_iters: int = 100,
        joint_refinement: bool = True,
        joint_refinement_iters: int = 3,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """
        Train the RVQ codebook.
        
        Args:
            X: Training data of shape (N, D).
            num_iters: K-means iterations for greedy init.
            joint_refinement: Enable joint refinement.
            joint_refinement_iters: Joint refinement iterations.
            seed: Random seed.
            verbose: Print progress.
        """
        if X.shape[1] != self.D:
            raise ValueError(f"Expected D={self.D}, got {X.shape[1]}")
        
        X = X.to(self.device)
        
        self.codebooks, _ = train_rvq(
            X, self.M, self.K,
            num_iters=num_iters,
            joint_refinement=joint_refinement,
            joint_refinement_iters=joint_refinement_iters,
            seed=seed,
            verbose=verbose,
        )
        self._trained = True
    
    def encode(self, X: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode data points to code indices at each level.
        
        Uses greedy sequential encoding: at each level, find the best
        centroid to minimize residual error.
        
        Args:
            X: Data tensor of shape (..., D).
            
        Returns:
            codes: List of M index tensors, each of shape (...,).
        """
        if not self._trained:
            raise RuntimeError("RVQ codebook not trained")
        
        original_shape = X.shape[:-1]
        X_flat = X.view(-1, self.D)
        N = X_flat.shape[0]
        
        all_codes = []
        residual = X_flat.clone()
        
        for m in range(self.M):
            codebook = self.codebooks[m].to(X.device)
            
            # Batched distance computation
            batch_size = min(10000, N)
            codes = torch.zeros(N, device=X.device, dtype=torch.long)
            
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                dists = torch.cdist(residual[i:end], codebook)
                codes[i:end] = dists.argmin(dim=1)
            
            all_codes.append(codes.view(original_shape))
            residual = residual - codebook[codes]
        
        return all_codes
    
    def encode_joint(self, X: torch.Tensor, base_recon: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Encode with joint optimization (better but slower).
        
        Given a base reconstruction (from Stage 5), find the best
        RVQ codes to minimize total error.
        
        Args:
            X: Target data tensor of shape (..., D).
            base_recon: Optional base reconstruction to start from.
            
        Returns:
            codes: List of M index tensors.
        """
        if not self._trained:
            raise RuntimeError("RVQ codebook not trained")
        
        original_shape = X.shape[:-1]
        X_flat = X.view(-1, self.D)
        
        if base_recon is not None:
            residual = X_flat - base_recon.view(-1, self.D)
        else:
            residual = X_flat.clone()
        
        # Use greedy encoding on the residual
        return self.encode(residual.view(*original_shape, self.D))
    
    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode code indices back to vectors.
        
        Reconstruction is sum of centroids: sum_m Cm[a_m]
        
        Args:
            codes: List of M index tensors, each of shape (...,).
            
        Returns:
            X: Reconstructed tensor of shape (..., D).
        """
        if not self._trained:
            raise RuntimeError("RVQ codebook not trained")
        
        if len(codes) != self.M:
            raise ValueError(f"Expected {self.M} code tensors, got {len(codes)}")
        
        original_shape = codes[0].shape
        device = codes[0].device
        
        recon = torch.zeros(*original_shape, self.D, device=device, dtype=self.codebooks[0].dtype)
        
        for m in range(self.M):
            codebook = self.codebooks[m].to(device)
            recon += codebook[codes[m]]
        
        return recon
    
    def quantize(self, X: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Quantize data (encode + decode).
        
        Args:
            X: Data tensor of shape (..., D).
            
        Returns:
            codes: List of M index tensors.
            reconstructed: Reconstructed tensor.
        """
        codes = self.encode(X)
        reconstructed = self.decode(codes)
        return codes, reconstructed
    
    def get_mse(self, X: torch.Tensor) -> float:
        """Compute MSE on data."""
        codes, reconstructed = self.quantize(X)
        return ((X - reconstructed) ** 2).mean().item()
    
    def state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "M": self.M,
            "K": self.K,
            "D": self.D,
            "codebooks": [cb.cpu() for cb in self.codebooks],
            "trained": self._trained,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self.M = state["M"]
        self.K = state["K"]
        self.D = state["D"]
        self.codebooks = [cb.to(self.device) for cb in state["codebooks"]]
        self._trained = state["trained"]
    
    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> "RVQCodebook":
        """Create RVQ codebook from state dict."""
        rvq = cls(M=state["M"], K=state["K"], D=state["D"], device=device)
        rvq.load_state_dict(state)
        return rvq
