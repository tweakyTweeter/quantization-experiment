"""
PGR-Q Configuration.

All experimental components are toggleable via config flags.
Default values follow the recommended configuration from the algorithm spec.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum


class ScalarFloorType(Enum):
    """Scalar floor quantization method."""
    BETA_MAX_LLOYD = "beta_max_lloyd"  # Recommended default
    DIGIT_PLANES = "digit_planes"  # Experimental p-adic style


class PrecisionMode(Enum):
    """Precision modes for mixed-precision allocation."""
    BASE = "base"
    BASE_PHASE = "base_phase"
    BASE_RESID = "base_resid"
    BASE_PHASE_RESID = "base_phase_resid"
    BASE_RESID_DIGITS = "base_resid_digits"


@dataclass
class PGRQConfig:
    """
    Configuration for PGR-Q quantization.
    
    Attributes:
        # Block and group structure
        block_size: Block dimension d (must be power of 2). Default 64.
        num_groups: Number of groups G per block. Default 4.
        
        # Base codebook
        base_codebook_size: K0, number of centroids in base codebook. Default 256.
        
        # Residual quantization
        num_residual_levels: M, number of additive residual codebook levels. Default 2.
        residual_codebook_size: K, centroids per residual level. Default 256.
        joint_refinement: Enable joint refinement after greedy RVQ init. Default True.
        joint_refinement_iters: Number of joint refinement iterations. Default 3.
        
        # Phase correction (EXPERIMENTAL - default OFF)
        phase_correction: Enable phase correction. Default False.
        phase_resolution: R, number of discrete phases (4, 8, or 16). Default 8.
        
        # Scalar floor
        scalar_floor_type: Type of scalar floor quantization. Default BETA_MAX_LLOYD.
        scalar_floor_bits: Bits per coordinate for scalar floor. Default 2.
        digit_planes_levels: L, bit-planes for digit-plane method. Default 4.
        
        # Mixed-precision allocation
        activation_aware_importance: Use activation-aware importance. Default True.
        target_bits_per_weight: Target average bits per weight. Default 4.0.
        
        # Codebook sharing
        cross_layer_codebook_sharing: Share codebooks across layers. Default True.
        per_group_codebooks: Use separate codebooks per group index. Default False.
        
        # Entropy coding
        entropy_coding: Use real entropy coding (vs mixed-radix packing). Default True.
        
        # Norm storage
        norm_dtype: Data type for storing norms. Default "float16".
        
        # Calibration
        calibration_samples: Number of calibration sequences. Default 128.
        calibration_seq_length: Sequence length for calibration. Default 2048.
        
        # What to quantize
        quantize_embeddings: Quantize embedding layer. Default False.
        quantize_lm_head: Quantize language model head. Default False.
        quantize_layernorms: Quantize layernorm parameters. Default False.
        
        # Device/compute
        device: Target device. Default "cuda" if available else "cpu".
        
        # K-means training
        kmeans_iters: Number of k-means iterations. Default 100.
        kmeans_init: K-means initialization method. Default "kmeans++".
        
        # Random seed
        seed: Random seed for reproducibility. Default 42.
    """
    
    # Block and group structure
    block_size: int = 64
    num_groups: int = 4
    
    # Base codebook
    base_codebook_size: int = 256
    
    # Residual quantization
    num_residual_levels: int = 2
    residual_codebook_size: int = 256
    joint_refinement: bool = True
    joint_refinement_iters: int = 3
    
    # Phase correction (EXPERIMENTAL - default OFF)
    phase_correction: bool = False
    phase_resolution: int = 8
    
    # Scalar floor
    scalar_floor_type: ScalarFloorType = ScalarFloorType.BETA_MAX_LLOYD
    scalar_floor_bits: int = 2
    digit_planes_levels: int = 4
    enable_scalar_floor: bool = False  # Only used in BASE_RESID_DIGITS mode
    
    # Mixed-precision allocation
    activation_aware_importance: bool = True
    target_bits_per_weight: float = 4.0
    
    # Codebook sharing
    cross_layer_codebook_sharing: bool = True
    per_group_codebooks: bool = False
    
    # Entropy coding
    entropy_coding: bool = True
    
    # Norm storage
    norm_dtype: str = "float16"
    
    # Calibration
    calibration_samples: int = 128
    calibration_seq_length: int = 2048
    calibration_dataset: str = "wikitext"  # "wikitext" or "c4"
    
    # What to quantize
    quantize_embeddings: bool = False
    quantize_lm_head: bool = False
    quantize_layernorms: bool = False
    
    # Device/compute
    device: str = "auto"
    
    # K-means training
    kmeans_iters: int = 100
    kmeans_init: str = "kmeans++"
    
    # Random seed  
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        # Check block_size is power of 2
        if self.block_size & (self.block_size - 1) != 0:
            raise ValueError(f"block_size must be power of 2, got {self.block_size}")
        
        # Check block_size is divisible by num_groups
        if self.block_size % self.num_groups != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must be divisible by num_groups ({self.num_groups})"
            )
        
        # Check group dimension is even (required for phase correction)
        group_dim = self.block_size // self.num_groups
        if group_dim % 2 != 0:
            raise ValueError(f"group_dim ({group_dim}) must be even for phase correction")
        
        # Check phase resolution is valid
        if self.phase_resolution not in [4, 8, 16]:
            raise ValueError(f"phase_resolution must be 4, 8, or 16, got {self.phase_resolution}")
        
        # Set device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def group_dim(self) -> int:
        """Dimension of each group (gd)."""
        return self.block_size // self.num_groups
    
    def get_bits_per_group(self, mode: PrecisionMode) -> float:
        """Calculate bits per group for a given precision mode."""
        import math
        
        base_bits = math.log2(self.base_codebook_size)  # log2(K0)
        
        if mode == PrecisionMode.BASE:
            return base_bits
        
        phase_bits = math.log2(self.phase_resolution) if self.phase_correction else 0
        resid_bits = self.num_residual_levels * math.log2(self.residual_codebook_size)
        scalar_bits = self.group_dim * self.scalar_floor_bits
        
        if mode == PrecisionMode.BASE_PHASE:
            return base_bits + phase_bits
        elif mode == PrecisionMode.BASE_RESID:
            return base_bits + resid_bits
        elif mode == PrecisionMode.BASE_PHASE_RESID:
            return base_bits + phase_bits + resid_bits
        elif mode == PrecisionMode.BASE_RESID_DIGITS:
            return base_bits + resid_bits + scalar_bits
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "block_size": self.block_size,
            "num_groups": self.num_groups,
            "base_codebook_size": self.base_codebook_size,
            "num_residual_levels": self.num_residual_levels,
            "residual_codebook_size": self.residual_codebook_size,
            "joint_refinement": self.joint_refinement,
            "joint_refinement_iters": self.joint_refinement_iters,
            "phase_correction": self.phase_correction,
            "phase_resolution": self.phase_resolution,
            "scalar_floor_type": self.scalar_floor_type.value,
            "scalar_floor_bits": self.scalar_floor_bits,
            "digit_planes_levels": self.digit_planes_levels,
            "enable_scalar_floor": self.enable_scalar_floor,
            "activation_aware_importance": self.activation_aware_importance,
            "target_bits_per_weight": self.target_bits_per_weight,
            "cross_layer_codebook_sharing": self.cross_layer_codebook_sharing,
            "per_group_codebooks": self.per_group_codebooks,
            "entropy_coding": self.entropy_coding,
            "norm_dtype": self.norm_dtype,
            "calibration_samples": self.calibration_samples,
            "calibration_seq_length": self.calibration_seq_length,
            "calibration_dataset": self.calibration_dataset,
            "quantize_embeddings": self.quantize_embeddings,
            "quantize_lm_head": self.quantize_lm_head,
            "quantize_layernorms": self.quantize_layernorms,
            "device": self.device,
            "kmeans_iters": self.kmeans_iters,
            "kmeans_init": self.kmeans_init,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PGRQConfig":
        """Create config from dictionary."""
        d = d.copy()
        if "scalar_floor_type" in d:
            d["scalar_floor_type"] = ScalarFloorType(d["scalar_floor_type"])
        return cls(**d)
