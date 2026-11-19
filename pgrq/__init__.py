"""
PGR-Q: Post-training Quantization for LLM Weights

A production-quality PTQ library implementing the PGR-Q algorithm with:
- Norm separation and unit direction quantization
- Random-signed Hadamard rotation (FWHT)
- Learned base codebooks with optional phase correction
- Residual quantization (greedy RVQ + joint refinement)
- Beta-Max-Lloyd scalar floor
- Mixed-precision allocation
- Real entropy coding
"""

from pgrq.core.config import PGRQConfig
from pgrq.core.quantizer import PGRQuantizer
from pgrq.hf.integration import quantize_model, load_quantized_model

__version__ = "0.1.0"
__all__ = [
    "PGRQConfig",
    "PGRQuantizer",
    "quantize_model",
    "load_quantized_model",
]
