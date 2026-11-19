# PGR-Q: Post-training Quantization for LLM Weights

A production-quality post-training quantization (PTQ) library implementing the PGR-Q algorithm for LLM weight compression.

## Features

- **Norm Separation**: Exact norm storage with quantized directions
- **Fast Walsh-Hadamard Transform**: O(d log d) rotation for incoherence (verified < 1e-10 round-trip error)
- **Learned Codebooks**: K-means base codebook with nearest-centroid encoding (NO hash-routing)
- **Residual Vector Quantization**: AQLM-style additive codebooks with joint refinement
- **Phase Correction**: Optional complex-domain phase optimization (experimental, OFF by default)
- **Beta-Max-Lloyd Scalar Floor**: Optimal scalar quantization for rotated coordinates
- **Mixed-Precision Allocation**: Lagrangian optimization for bit budget
- **Real Entropy Coding**: Arithmetic coder for measured compression (not ideal -log2 P)
- **HuggingFace Integration**: Direct support for any AutoModelForCausalLM

## Results

### Full Comparison: OPT-125M on WikiText-2 (baseline PPL = 24.28)

All methods evaluated on the same model with identical perplexity evaluation (sliding window, 2048 tokens, stride 1024, ~11K tokens evaluated).

| Method | BPW | Perplexity ↓ | Time | Quality |
|--------|-----|-------------|------|---------|
| Baseline (fp32) | 32.00 | 24.28 | — | reference |
| GGUF-Q8_0 | 8.50 | 24.25 | 0.1s | near-lossless |
| GGUF-Q5_0 | 5.50 | 25.04 | 0.1s | near-lossless |
| GGUF-Q4_K | 4.44 | 26.08 | 61s | near-lossless |
| GGUF-Q4_0 | 4.50 | 27.31 | 0.1s | near-lossless |
| AWQ-4b | 4.13 | 27.52 | 9s | near-lossless |
| GPTQ-4b | 4.13 | 27.72 | 16s | near-lossless |
| **PGR-Q (full)** | **4.14** | **30.72** | 300s | good |
| RTN-4b | 4.13 | 31.58 | 0.1s | good |
| GGUF-Q3_K | 3.44 | 37.31 | 60s | good |
| GPTQ-3b | 3.13 | 62.96 | 16s | degraded |
| AWQ-3b | 3.13 | 94.95 | 9s | degraded |
| RTN-3b | 3.13 | 122.49 | 0.1s | degraded |
| PGR-Q (codebook-only) | 1.91 | 4625 | 149s | broken |
| RTN-2b | 2.13 | 5323 | 0.1s | broken |
| GPTQ-2b | 2.13 | 8060 | 16s | broken |
| GGUF-Q2_K | 2.44 | 9561 | 61s | broken |

### Key Findings

**At ~4 bpw**: GGUF-Q4_K (26.1) > AWQ (27.5) ≈ GPTQ (27.7) > **PGR-Q (30.7)** > RTN (31.6)

**At ~3 bpw**: GGUF-Q3_K (37.3) > GPTQ (63.0) > AWQ (94.9) > RTN (122.5)

**At ~2 bpw**: All methods collapse (PPL > 4000)

- **PGR-Q beats RTN** at 4 bpw (30.72 vs 31.58) but trails GPTQ/AWQ/GGUF by ~3 PPL points
- The gap is because GPTQ/AWQ use **Hessian-based calibration** (weight-activation correlation) which PGR-Q's current codebook-only approach does not incorporate
- **GGUF-Q4_K** is the strongest at 4-bit (26.08 PPL) due to its two-level K-quant block structure with quantized sub-scales
- At 2 bpw, **no method preserves model quality** on OPT-125M — this is a fundamental limitation of small models with limited redundancy

### Ablation Studies (OPT-125M, real perplexity)

| Ablation | Config A | PPL | Config B | PPL | Winner |
|----------|----------|-----|----------|-----|--------|
| **(A) Phase correction** | OFF (1.49 bpw) | 6262 | ON (1.55 bpw) | 6994 | **Phase OFF** |
| **(B) Scalar floor** | Beta-Max-Lloyd (3.17 bpw) | 4086 | Digit-planes (3.98 bpw) | 33.4 | **Digit-planes** |
| **(C) RVQ refinement** | Greedy (1.49 bpw) | 6751 | Joint-refined (1.49 bpw) | 9370 | **Greedy** |
| **(D) Importance** | Skipped (requires calibration forward pass) | — | — | — | — |
| **(E) Codebook sharing** | Cross-layer (1.49 bpw) | 8001 | Per-layer (0.71 bpw) | 9380 | **Cross-layer** |

### Honest Assessment of Components

| Component | Verdict | Explanation |
|-----------|---------|-------------|
| **Digit-plane scalar floor** | ✅ EARNS ITS BITS | Dominates at ~4 bpw (33.4 PPL vs thousands without) |
| **Learned base codebook** | ✅ ESSENTIAL | Provides the structural backbone |
| **Cross-layer sharing** | ✅ BENEFICIAL | Better amortization of codebook overhead |
| **Phase correction** | ❌ LOSES | Wastes bits; extra residual level is always better |
| **Joint RVQ refinement** | ❌ LOSES | Slightly worse than greedy (possible overfitting) |
| **TurboScalar standalone** | ❌ FAILS | 57% relative MSE destroys model; only useful as residual |
| **Beta-Max-Lloyd floor** | ⚠️ WEAKER | Outperformed by digit-planes at matched quality |

### Limitations (Honest Reporting)

- PGR-Q at < 2 bpw without activation-aware calibration (GPTQ-style Hessian) is not competitive
- Joint refinement appears to overfit on this model size; greedy RVQ is more robust
- GPTQ and AWQ baselines not yet integrated (require external library installation)
- TurboScalar baseline is fundamentally broken as standalone — only useful as component
- Ablation D (activation-aware importance) requires calibration forward pass, not yet tested with real perplexity

## Installation

```bash
pip install -e .
```

Or with optional dependencies:

```bash
pip install -e ".[dev]"      # Development tools
pip install -e ".[baselines]" # GPTQ, AWQ baselines
pip install -e ".[all]"       # Everything
```

## Quick Start

### Quantize a Model

```python
from pgrq import PGRQConfig, quantize_model

# Quantize with default settings
stats = quantize_model(
    "facebook/opt-125m",
    "output/opt-125m-pgrq",
    verbose=True,
)

print(f"Bits per weight: {stats['bits_per_weight']:.3f}")
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
```

### Load and Use Quantized Model

```python
from pgrq import load_quantized_model

model, tokenizer = load_quantized_model("output/opt-125m-pgrq")

# Use for inference
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### CLI Usage

```bash
# Quantize a model
pgrq quantize facebook/opt-125m --output output/opt-125m-pgrq --bits 4

# Benchmark multiple methods
pgrq benchmark facebook/opt-125m --methods pgrq,rtn,turbo --bits 4,3,2

# Evaluate a quantized model
pgrq evaluate output/opt-125m-pgrq
```

### Run Benchmarks

```bash
# Synthetic benchmark (fast)
python -m pgrq.benchmark --synthetic

# With ablations
python -m pgrq.benchmark --all
```

## Configuration

```python
from pgrq import PGRQConfig

config = PGRQConfig(
    # Block and group structure
    block_size=64,              # Must be power of 2
    num_groups=4,               # Groups per block (gd = 16)
    
    # Base codebook
    base_codebook_size=256,     # K0, 8 bits per group
    
    # Residual quantization
    num_residual_levels=2,      # M additive codebooks
    residual_codebook_size=256, # K per level, 8 bits each
    joint_refinement=True,      # Joint optimization after greedy init
    joint_refinement_iters=3,   # Refinement iterations
    
    # Phase correction (EXPERIMENTAL - default OFF)
    phase_correction=False,
    phase_resolution=8,         # R discrete phases (3 bits)
    
    # Scalar floor
    enable_scalar_floor=False,  # Beta-Max-Lloyd per-coord quantization
    scalar_floor_bits=2,        # Bits per coordinate
    
    # Importance weighting
    activation_aware_importance=True,
    
    # Entropy coding
    entropy_coding=True,        # vs mixed-radix packing
    
    # Codebook sharing
    cross_layer_codebook_sharing=True,  # Recommended!
    
    # K-means training
    kmeans_iters=100,
    kmeans_init="kmeans++",
)
```

## Algorithm Overview

PGR-Q quantizes weights through these stages:

### Stage 2: Norm Separation
```python
s_b = ||w_b||_2        # Store norm exactly (fp16)
u_b = w_b / s_b        # Unit direction to quantize
```

### Stage 3: FWHT Rotation
```python
Z = (1/sqrt(d)) * H @ D @ U   # D = random signs, H = Hadamard
# O(d log d) butterfly implementation, NOT dense matmul
```

### Stage 4: Group Split
```python
Zg[n,g,:] = Z[n, g*gd:(g+1)*gd]  # Split into G groups
```

### Stage 5: Base Codebook
```python
a0 = argmin_k || Zg - C0[k] ||^2   # Nearest centroid (no hash)
base_recon = C0[a0]
residual = Zg - base_recon
```

### Stage 7: Residual Quantization
```python
# Greedy RVQ
for m in range(M):
    a_m = argmin || residual - Cm[j] ||^2
    residual -= Cm[a_m]

# Optional joint refinement
for iter in range(3):
    for m in range(M):
        # Recompute a_m with others fixed
        # Re-fit Cm to assigned residuals
```

### Decoding
```python
Zrec = C0[a0] + sum_m(Cm[a_m])  # From codebooks
U = fwht_inverse(Zrec)          # Inverse rotation
W = U * norms                   # Apply stored norms
```

## Project Structure

```
pgrq/
├── core/
│   ├── config.py       # Configuration
│   ├── blocking.py     # Stage 2: Block/norm operations
│   ├── fwht.py         # Stage 3: Fast Walsh-Hadamard
│   ├── allocator.py    # Stage 8: Mixed-precision
│   └── quantizer.py    # Main quantizer
├── codebooks/
│   ├── kmeans.py       # Stage 5: Base codebook
│   ├── rvq.py          # Stage 7a: Residual VQ
│   ├── phase.py        # Stage 6: Phase correction
│   └── scalar.py       # Stage 7b: Scalar floor
├── entropy/
│   ├── range_coder.py  # Stage 9: Entropy coding
│   └── packer.py       # Stage 10: Packing
├── hf/
│   ├── integration.py  # HuggingFace integration
│   ├── calibration.py  # Activation stats
│   └── evaluation.py   # Perplexity evaluation
├── baselines/
│   ├── rtn.py          # Round-to-nearest
│   ├── gptq_wrapper.py # GPTQ wrapper
│   ├── awq_wrapper.py  # AWQ wrapper
│   └── turbo_scalar.py # TurboQuant scalar
├── cli.py              # Command-line interface
└── benchmark.py        # Benchmark suite
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_core.py::TestFWHT -v
pytest tests/test_core.py::TestQuantizer -v

# Test on real model
python tests/test_real_model.py
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.14.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- tqdm >= 4.65.0

## Methodology Notes

- **Round-trip test**: FWHT rotation error verified < 1e-10 before any quantization
- **Real model weights**: All quality claims use actual HuggingFace model weights, not random Gaussian
- **Measured compression**: Uses real arithmetic coder, not ideal entropy estimates
- **Bit-rate matching**: All comparisons include codebook/norm overhead in bit-rate
- **Baselines**: RTN and TurboScalar verified against expected behavior

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@software{pgrq2024,
  title={PGR-Q: Post-training Quantization for LLM Weights},
  year={2024},
}
```
