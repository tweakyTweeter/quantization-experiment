"""
Comprehensive unit tests for PGR-Q.

Tests each stage in isolation and verifies the full pipeline.
"""

import pytest
import torch
import numpy as np


class TestBlocking:
    """Tests for Stage 2: Block reshaping and norm separation."""
    
    def test_reshape_to_blocks_basic(self):
        """Test basic block reshaping."""
        from pgrq.core.blocking import reshape_to_blocks, reshape_from_blocks
        
        W = torch.randn(128, 256)
        block_size = 64
        
        blocks, shape_info = reshape_to_blocks(W, block_size)
        
        assert blocks.shape[1] == block_size
        assert shape_info["original_shape"] == W.shape
        assert shape_info["block_size"] == block_size
    
    def test_reshape_roundtrip(self):
        """Test that reshape is invertible."""
        from pgrq.core.blocking import reshape_to_blocks, reshape_from_blocks
        
        W = torch.randn(100, 200)  # Not divisible by 64
        block_size = 64
        
        blocks, shape_info = reshape_to_blocks(W, block_size)
        W_rec = reshape_from_blocks(blocks, shape_info)
        
        assert W_rec.shape == W.shape
        assert torch.allclose(W_rec, W, atol=1e-6)
    
    def test_norm_extraction(self):
        """Test norm extraction produces unit vectors."""
        from pgrq.core.blocking import extract_norms
        
        blocks = torch.randn(100, 64)
        unit_dirs, norms = extract_norms(blocks)
        
        # Check unit vectors have norm 1
        computed_norms = torch.norm(unit_dirs, dim=1)
        assert torch.allclose(computed_norms, torch.ones_like(computed_norms), atol=1e-5)
        
        # Check norms are correct (fp16 has limited precision)
        expected_norms = torch.norm(blocks, dim=1, keepdim=True)
        assert torch.allclose(norms.float(), expected_norms.float(), atol=1e-2)
    
    def test_norm_apply_roundtrip(self):
        """Test that apply_norms inverts extract_norms."""
        from pgrq.core.blocking import extract_norms, apply_norms
        
        blocks = torch.randn(100, 64)
        unit_dirs, norms = extract_norms(blocks)
        blocks_rec = apply_norms(unit_dirs, norms)
        
        # fp16 norms have limited precision
        assert torch.allclose(blocks_rec, blocks, atol=1e-2)
    
    def test_group_split_merge(self):
        """Test group splitting and merging."""
        from pgrq.core.blocking import split_into_groups, merge_groups
        
        Z = torch.randn(100, 64)
        num_groups = 4
        
        Zg = split_into_groups(Z, num_groups)
        
        assert Zg.shape == (100, 4, 16)
        
        Z_rec = merge_groups(Zg)
        assert torch.allclose(Z_rec, Z)


class TestFWHT:
    """Tests for Stage 3: Fast Walsh-Hadamard Transform."""
    
    def test_fwht_basic(self):
        """Test basic FWHT computation."""
        from pgrq.core.fwht import fwht
        
        X = torch.randn(10, 64)
        Y = fwht(X)
        
        assert Y.shape == X.shape
    
    def test_fwht_inverse(self):
        """Test FWHT is its own inverse (when normalized)."""
        from pgrq.core.fwht import fwht, fwht_inverse
        
        X = torch.randn(10, 64)
        Y = fwht(X)
        X_rec = fwht_inverse(Y)
        
        assert torch.allclose(X_rec, X, atol=1e-5)
    
    def test_fwht_rotate_roundtrip(self):
        """Test FWHT rotation round-trip error < 1e-10."""
        from pgrq.core.fwht import fwht_rotate, fwht_rotate_inverse, verify_fwht_roundtrip
        
        U = torch.randn(100, 64, dtype=torch.float64)  # Double precision for tight tolerance
        # Normalize to unit vectors
        U = U / torch.norm(U, dim=1, keepdim=True)
        
        seed = 42
        
        passed, max_error = verify_fwht_roundtrip(U, seed, tol=1e-10)
        
        assert passed, f"Round-trip error {max_error} exceeds 1e-10"
        assert max_error < 1e-10
    
    def test_fwht_different_seeds(self):
        """Test that different seeds give different rotations."""
        from pgrq.core.fwht import fwht_rotate
        
        U = torch.randn(10, 64)
        
        Z1 = fwht_rotate(U, seed=1)
        Z2 = fwht_rotate(U, seed=2)
        
        assert not torch.allclose(Z1, Z2)
    
    def test_fwht_preserves_norm(self):
        """Test that FWHT rotation preserves L2 norm."""
        from pgrq.core.fwht import fwht_rotate
        
        U = torch.randn(100, 64)
        norms_before = torch.norm(U, dim=1)
        
        Z = fwht_rotate(U, seed=42)
        norms_after = torch.norm(Z, dim=1)
        
        assert torch.allclose(norms_before, norms_after, atol=1e-5)


class TestKMeansCodebook:
    """Tests for Stage 5: Base codebook."""
    
    def test_kmeans_training(self):
        """Test K-means codebook training."""
        from pgrq.codebooks.kmeans import KMeansCodebook
        
        X = torch.randn(1000, 16)
        K = 256
        
        codebook = KMeansCodebook(K=K, D=16)
        codebook.train(X, num_iters=50, verbose=False)
        
        assert codebook.centroids.shape == (K, 16)
        assert codebook._trained
    
    def test_kmeans_encode_decode(self):
        """Test encoding and decoding."""
        from pgrq.codebooks.kmeans import KMeansCodebook
        
        X = torch.randn(1000, 16)
        K = 64
        
        codebook = KMeansCodebook(K=K, D=16)
        codebook.train(X, num_iters=50, verbose=False)
        
        codes = codebook.encode(X)
        assert codes.shape == (1000,)
        assert codes.min() >= 0
        assert codes.max() < K
        
        X_rec = codebook.decode(codes)
        assert X_rec.shape == X.shape
    
    def test_kmeans_reduces_mse(self):
        """Test that K-means reduces MSE compared to random."""
        from pgrq.codebooks.kmeans import KMeansCodebook
        
        X = torch.randn(1000, 16)
        K = 64
        
        codebook = KMeansCodebook(K=K, D=16)
        codebook.train(X, num_iters=100, verbose=False)
        
        mse = codebook.get_mse(X)
        
        # MSE should be significantly less than variance of data
        variance = X.var().item()
        # With K=64 and D=16, we should get at least 40% reduction
        assert mse < variance * 0.7  # Should reduce MSE


class TestRVQ:
    """Tests for Stage 7a: Residual Vector Quantization."""
    
    def test_rvq_training(self):
        """Test RVQ training."""
        from pgrq.codebooks.rvq import RVQCodebook
        
        X = torch.randn(1000, 16)
        M = 2
        K = 64
        
        rvq = RVQCodebook(M=M, K=K, D=16)
        rvq.train(X, num_iters=30, joint_refinement=True, verbose=False)
        
        assert len(rvq.codebooks) == M
        assert all(cb.shape == (K, 16) for cb in rvq.codebooks)
    
    def test_rvq_encode_decode(self):
        """Test RVQ encoding and decoding."""
        from pgrq.codebooks.rvq import RVQCodebook
        
        X = torch.randn(500, 16)
        M = 2
        K = 64
        
        rvq = RVQCodebook(M=M, K=K, D=16)
        rvq.train(X, num_iters=30, verbose=False)
        
        codes = rvq.encode(X)
        assert len(codes) == M
        assert all(c.shape == (500,) for c in codes)
        
        X_rec = rvq.decode(codes)
        assert X_rec.shape == X.shape
    
    def test_rvq_improves_over_base(self):
        """Test that RVQ improves over single codebook."""
        from pgrq.codebooks.kmeans import KMeansCodebook
        from pgrq.codebooks.rvq import RVQCodebook
        
        X = torch.randn(1000, 16)
        K = 64
        
        # Base codebook MSE
        base = KMeansCodebook(K=K, D=16)
        base.train(X, num_iters=50, verbose=False)
        base_mse = base.get_mse(X)
        
        # RVQ MSE (using equivalent bits: M=2 with K=64 each)
        rvq = RVQCodebook(M=2, K=K, D=16)
        rvq.train(X, num_iters=50, joint_refinement=True, verbose=False)
        rvq_mse = rvq.get_mse(X)
        
        assert rvq_mse < base_mse


class TestPhaseCorrection:
    """Tests for Stage 6: Phase correction."""
    
    def test_complex_conversion(self):
        """Test real-to-complex conversion."""
        from pgrq.codebooks.phase import to_complex, to_real
        
        v = torch.randn(10, 16)
        z = to_complex(v)
        
        assert z.shape == (10, 8)
        
        v_rec = to_real(z)
        assert torch.allclose(v_rec, v)
    
    def test_phase_correction_improves(self):
        """Test that phase correction can improve reconstruction."""
        from pgrq.codebooks.phase import PhaseCorrector
        
        corrector = PhaseCorrector(R=8)
        
        target = torch.randn(100, 16)
        centroid = target + 0.1 * torch.randn_like(target)  # Noisy version
        
        mse_before, mse_after = corrector.get_improvement(target, centroid)
        
        # Phase correction should help (or at least not hurt)
        assert mse_after <= mse_before + 1e-6


class TestScalarQuantizers:
    """Tests for Stage 7b: Scalar floor quantization."""
    
    def test_beta_max_lloyd_quantizer(self):
        """Test Beta-Max-Lloyd quantizer."""
        from pgrq.codebooks.scalar import BetaMaxLloydQuantizer
        
        quantizer = BetaMaxLloydQuantizer(bits=2, block_size=64)
        
        # Generate data matching expected distribution
        X = torch.randn(1000) / 8  # Roughly matches post-Hadamard variance
        
        codes, X_rec = quantizer.quantize(X)
        
        assert codes.shape == X.shape
        assert codes.min() >= 0
        assert codes.max() < 4  # 2 bits = 4 levels
    
    def test_digit_plane_quantizer(self):
        """Test digit-plane quantizer."""
        from pgrq.codebooks.scalar import DigitPlaneQuantizer
        
        quantizer = DigitPlaneQuantizer(levels=2, block_size=64)
        
        X = torch.randn(1000) / 8
        
        codes, X_rec = quantizer.quantize(X)
        
        assert codes.shape == X.shape
        assert codes.min() >= 0
        assert codes.max() < 4  # 2 bits


class TestEntropyCoding:
    """Tests for Stage 9: Entropy coding."""
    
    def test_range_coder_roundtrip(self):
        """Test range coder encode/decode roundtrip."""
        from pgrq.entropy.range_coder import encode_symbols, decode_symbols
        
        # Create symbols with non-uniform distribution
        symbols = torch.randint(0, 256, (1000,))
        symbols[:500] = torch.randint(0, 10, (500,))  # Skewed distribution
        
        data, freqs = encode_symbols(symbols, num_symbols=256)
        decoded = decode_symbols(data, freqs, num_symbols=1000)
        
        assert torch.equal(symbols, decoded)
    
    def test_entropy_compression(self):
        """Test that entropy coding achieves compression."""
        from pgrq.entropy.range_coder import encode_symbols, compute_ideal_bits
        
        # Highly skewed distribution should compress well
        symbols = torch.zeros(1000, dtype=torch.long)
        symbols[:10] = torch.arange(10)  # Mostly zeros
        
        data, freqs = encode_symbols(symbols, num_symbols=256)
        
        # Compressed size should be much less than 1 byte per symbol
        bits_per_symbol = len(data) * 8 / 1000
        assert bits_per_symbol < 1.0


class TestPacker:
    """Tests for Stage 10: Packing."""
    
    def test_mixed_radix_packer(self):
        """Test mixed-radix integer packing."""
        from pgrq.entropy.packer import MixedRadixPacker
        
        radices = [256, 256, 8]  # Base, residual, phase
        packer = MixedRadixPacker(radices)
        
        values = [
            torch.randint(0, 256, (100,)),
            torch.randint(0, 256, (100,)),
            torch.randint(0, 8, (100,)),
        ]
        
        packed = packer.pack(values)
        unpacked = packer.unpack(packed)
        
        for orig, rec in zip(values, unpacked):
            assert torch.equal(orig, rec)
    
    def test_entropy_packer_roundtrip(self):
        """Test entropy packer encode/decode roundtrip."""
        from pgrq.entropy.packer import EntropyPacker
        
        packer = EntropyPacker(
            base_codebook_size=256,
            residual_codebook_size=64,
            num_residual_levels=2,
            use_entropy_coding=True,
        )
        
        base_codes = torch.randint(0, 256, (1000,))
        residual_codes = [
            torch.randint(0, 64, (1000,)),
            torch.randint(0, 64, (1000,)),
        ]
        norms = torch.randn(1000, 1).half()
        
        packed = packer.pack(
            base_codes=base_codes,
            residual_codes=residual_codes,
            norms=norms,
        )
        
        unpacked = packer.unpack(packed)
        
        assert torch.equal(unpacked["base_codes"], base_codes)
        assert torch.allclose(unpacked["norms"], norms.cpu())


class TestQuantizer:
    """Tests for the main PGR-Q quantizer."""
    
    def test_quantizer_train(self):
        """Test quantizer codebook training."""
        from pgrq.core.config import PGRQConfig
        from pgrq.core.quantizer import PGRQuantizer
        
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=64,
            num_residual_levels=1,
            residual_codebook_size=32,
            kmeans_iters=30,
        )
        
        quantizer = PGRQuantizer(config)
        
        weights = [torch.randn(128, 256) for _ in range(3)]
        quantizer.train_codebooks(weights, verbose=False)
        
        assert quantizer._codebooks_trained
    
    def test_quantizer_roundtrip(self):
        """Test quantize/dequantize roundtrip."""
        from pgrq.core.config import PGRQConfig
        from pgrq.core.quantizer import PGRQuantizer
        
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=64,
            num_residual_levels=1,
            residual_codebook_size=32,
            kmeans_iters=30,
            entropy_coding=False,  # Simpler for testing
        )
        
        quantizer = PGRQuantizer(config)
        
        W = torch.randn(128, 256)
        quantizer.train_codebooks([W], verbose=False)
        
        quantized = quantizer.quantize(W, verbose=False)
        W_rec = quantizer.dequantize(quantized)
        
        assert W_rec.shape == W.shape
        
        # Check reconstruction error is reasonable
        mse = ((W - W_rec) ** 2).mean().item()
        variance = W.var().item()
        assert mse < variance * 0.5  # Should preserve most information


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline_synthetic(self):
        """Test full pipeline on synthetic weights."""
        from pgrq.core.config import PGRQConfig
        from pgrq.core.quantizer import PGRQuantizer
        
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=128,
            num_residual_levels=2,
            residual_codebook_size=64,
            joint_refinement=True,
            kmeans_iters=50,
            entropy_coding=True,
        )
        
        quantizer = PGRQuantizer(config)
        
        # Simulate multiple linear layers
        weights = [
            torch.randn(256, 512),
            torch.randn(512, 256),
            torch.randn(256, 256),
        ]
        
        # Train
        quantizer.train_codebooks(weights, verbose=False)
        
        # Quantize each weight
        quantized_weights = []
        for W in weights:
            qt = quantizer.quantize(W, verbose=False)
            quantized_weights.append(qt)
        
        # Dequantize and check
        for W_orig, qt in zip(weights, quantized_weights):
            W_rec = quantizer.dequantize(qt)
            
            assert W_rec.shape == W_orig.shape
            
            mse = ((W_orig - W_rec) ** 2).mean().item()
            variance = W_orig.var().item()
            
            # Reconstruction should preserve significant information
            assert mse < variance * 0.3, f"MSE {mse} too high vs variance {variance}"
    
    def test_compression_stats(self):
        """Test compression statistics computation."""
        from pgrq.core.config import PGRQConfig
        from pgrq.core.quantizer import PGRQuantizer
        
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=256,
            num_residual_levels=2,
            residual_codebook_size=256,
            kmeans_iters=30,
        )
        
        quantizer = PGRQuantizer(config)
        
        W = torch.randn(512, 1024)
        quantizer.train_codebooks([W], verbose=False)
        
        quantized = quantizer.quantize(W, verbose=False)
        stats = quantizer.get_compression_stats(quantized)
        
        assert "bits_per_weight" in stats
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] > 1  # Should compress


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
