#!/usr/bin/env python
"""
Benchmark script for PGR-Q quantization.

Runs comprehensive benchmarks comparing PGR-Q against baselines.
"""

import argparse
import json
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

from pgrq.core.config import PGRQConfig
from pgrq.core.quantizer import PGRQuantizer


def benchmark_pgrq_synthetic():
    """Quick synthetic benchmark to verify the pipeline works."""
    print("=" * 60)
    print("PGR-Q Synthetic Benchmark")
    print("=" * 60)
    
    configs = [
        {"name": "2-bit", "base_k": 64, "resid_levels": 1, "resid_k": 32},
        {"name": "4-bit", "base_k": 256, "resid_levels": 2, "resid_k": 256},
    ]
    
    # Create synthetic weights (simulating a small model)
    weights = [
        torch.randn(256, 512),   # ~130K params
        torch.randn(512, 256),   # ~130K params
        torch.randn(256, 256),   # ~65K params
        torch.randn(512, 512),   # ~260K params
    ]
    
    total_params = sum(w.numel() for w in weights)
    print(f"Total parameters: {total_params:,}")
    print()
    
    results = []
    
    for cfg_dict in configs:
        print(f"\nTesting {cfg_dict['name']} configuration...")
        
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=cfg_dict["base_k"],
            num_residual_levels=cfg_dict["resid_levels"],
            residual_codebook_size=cfg_dict["resid_k"],
            joint_refinement=True,
            kmeans_iters=50,
            entropy_coding=True,
        )
        
        quantizer = PGRQuantizer(config)
        
        # Time training
        start = time.time()
        quantizer.train_codebooks(weights, verbose=False)
        train_time = time.time() - start
        
        # Time quantization and get stats
        start = time.time()
        total_bytes = 0
        total_mse = 0.0
        
        for w in weights:
            qt = quantizer.quantize(w, verbose=False)
            stats = quantizer.get_compression_stats(qt)
            total_bytes += stats["total_bytes"]
            
            # Check reconstruction error
            w_rec = quantizer.dequantize(qt)
            mse = ((w - w_rec) ** 2).mean().item()
            total_mse += mse * w.numel()
        
        quant_time = time.time() - start
        
        bits_per_weight = total_bytes * 8 / total_params
        avg_mse = total_mse / total_params
        
        result = {
            "config": cfg_dict["name"],
            "bits_per_weight": bits_per_weight,
            "mse": avg_mse,
            "train_time_s": train_time,
            "quant_time_s": quant_time,
        }
        results.append(result)
        
        print(f"  Bits per weight: {bits_per_weight:.3f}")
        print(f"  MSE: {avg_mse:.6f}")
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Quant time: {quant_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Config':<15} {'BPW':>10} {'MSE':>12} {'Time':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['config']:<15} {r['bits_per_weight']:>10.3f} {r['mse']:>12.6f} {r['train_time_s'] + r['quant_time_s']:>10.2f}s")
    
    return results


def benchmark_rtn_synthetic():
    """Quick RTN baseline on synthetic data."""
    from pgrq.baselines.rtn import RTNQuantizer
    
    print("\n" + "=" * 60)
    print("RTN Baseline Synthetic Benchmark")
    print("=" * 60)
    
    weights = [
        torch.randn(256, 512),
        torch.randn(512, 256),
        torch.randn(256, 256),
        torch.randn(512, 512),
    ]
    
    total_params = sum(w.numel() for w in weights)
    results = []
    
    for bits in [4, 3, 2]:
        print(f"\nTesting {bits}-bit RTN...")
        
        quantizer = RTNQuantizer(bits=bits, group_size=128)
        
        total_mse = 0.0
        total_bytes = 0
        
        for w in weights:
            qt, scales, zeros = quantizer.quantize(w)
            w_rec = quantizer.dequantize(qt, scales, zeros, w.shape)
            
            mse = ((w - w_rec) ** 2).mean().item()
            total_mse += mse * w.numel()
            
            sizes = quantizer.get_compressed_size(w.shape)
            total_bytes += sizes["total_bytes"]
        
        bits_per_weight = total_bytes * 8 / total_params
        avg_mse = total_mse / total_params
        
        result = {
            "config": f"RTN-{bits}b",
            "bits_per_weight": bits_per_weight,
            "mse": avg_mse,
        }
        results.append(result)
        
        print(f"  Bits per weight: {bits_per_weight:.3f}")
        print(f"  MSE: {avg_mse:.6f}")
    
    return results


def benchmark_turbo_scalar_synthetic():
    """Quick TurboQuant scalar baseline on synthetic data."""
    from pgrq.baselines.turbo_scalar import TurboScalarQuantizer
    
    print("\n" + "=" * 60)
    print("TurboQuant Scalar Baseline Synthetic Benchmark")
    print("=" * 60)
    
    weights = [
        torch.randn(256, 512),
        torch.randn(512, 256),
        torch.randn(256, 256),
        torch.randn(512, 512),
    ]
    
    total_params = sum(w.numel() for w in weights)
    results = []
    
    for bits in [4, 3, 2]:
        print(f"\nTesting {bits}-bit TurboScalar...")
        
        quantizer = TurboScalarQuantizer(bits=bits, block_size=64)
        
        total_mse = 0.0
        total_bytes = 0
        
        for w in weights:
            codes, norms, shape_info = quantizer.quantize(w)
            w_rec = quantizer.dequantize(codes, norms, shape_info)
            
            mse = ((w - w_rec) ** 2).mean().item()
            total_mse += mse * w.numel()
            
            sizes = quantizer.get_compressed_size(w.shape)
            total_bytes += sizes["total_bytes"]
        
        bits_per_weight = total_bytes * 8 / total_params
        avg_mse = total_mse / total_params
        
        result = {
            "config": f"TurboScalar-{bits}b",
            "bits_per_weight": bits_per_weight,
            "mse": avg_mse,
        }
        results.append(result)
        
        print(f"  Bits per weight: {bits_per_weight:.3f}")
        print(f"  MSE: {avg_mse:.6f}")
    
    return results


def run_ablation_synthetic():
    """Run ablation studies on synthetic data."""
    print("\n" + "=" * 60)
    print("Ablation Studies (Synthetic)")
    print("=" * 60)
    
    weights = [torch.randn(256, 512) for _ in range(4)]
    total_params = sum(w.numel() for w in weights)
    
    ablations = {}
    
    # A. Phase correction ON vs OFF
    print("\n[A] Phase correction ON vs OFF")
    for phase_on in [False, True]:
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=128,
            num_residual_levels=2,
            residual_codebook_size=64,
            phase_correction=phase_on,
            kmeans_iters=30,
        )
        
        quantizer = PGRQuantizer(config)
        quantizer.train_codebooks(weights, verbose=False)
        
        total_mse = 0.0
        for w in weights:
            qt = quantizer.quantize(w, verbose=False)
            w_rec = quantizer.dequantize(qt)
            total_mse += ((w - w_rec) ** 2).sum().item()
        
        avg_mse = total_mse / total_params
        label = "Phase ON" if phase_on else "Phase OFF"
        print(f"  {label}: MSE = {avg_mse:.6f}")
        ablations[f"A_{label}"] = avg_mse
    
    # C. Greedy RVQ vs Joint-refined
    print("\n[C] Greedy RVQ vs Joint-refined")
    for joint in [False, True]:
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=128,
            num_residual_levels=2,
            residual_codebook_size=64,
            joint_refinement=joint,
            kmeans_iters=30,
        )
        
        quantizer = PGRQuantizer(config)
        quantizer.train_codebooks(weights, verbose=False)
        
        total_mse = 0.0
        for w in weights:
            qt = quantizer.quantize(w, verbose=False)
            w_rec = quantizer.dequantize(qt)
            total_mse += ((w - w_rec) ** 2).sum().item()
        
        avg_mse = total_mse / total_params
        label = "Joint" if joint else "Greedy"
        print(f"  {label}: MSE = {avg_mse:.6f}")
        ablations[f"C_{label}"] = avg_mse
    
    # E. Per-layer vs cross-layer codebooks
    print("\n[E] Per-layer vs Cross-layer codebooks")
    for shared in [False, True]:
        config = PGRQConfig(
            block_size=64,
            num_groups=4,
            base_codebook_size=128,
            num_residual_levels=1,
            residual_codebook_size=64,
            cross_layer_codebook_sharing=shared,
            kmeans_iters=30,
        )
        
        quantizer = PGRQuantizer(config)
        quantizer.train_codebooks(weights, verbose=False)
        
        total_mse = 0.0
        for w in weights:
            qt = quantizer.quantize(w, verbose=False)
            w_rec = quantizer.dequantize(qt)
            total_mse += ((w - w_rec) ** 2).sum().item()
        
        avg_mse = total_mse / total_params
        label = "Cross-layer" if shared else "Per-layer"
        print(f"  {label}: MSE = {avg_mse:.6f}")
        ablations[f"E_{label}"] = avg_mse
    
    return ablations


def main():
    parser = argparse.ArgumentParser(description="PGR-Q Benchmark Suite")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic benchmarks")
    parser.add_argument("--ablations", action="store_true", help="Run ablation studies")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--output", "-o", type=str, help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    if not (args.synthetic or args.ablations or args.all):
        args.synthetic = True  # Default to synthetic
    
    results = {}
    
    if args.synthetic or args.all:
        pgrq_results = benchmark_pgrq_synthetic()
        results["pgrq"] = pgrq_results
        
        rtn_results = benchmark_rtn_synthetic()
        results["rtn"] = rtn_results
        
        turbo_results = benchmark_turbo_scalar_synthetic()
        results["turbo_scalar"] = turbo_results
    
    if args.ablations or args.all:
        ablation_results = run_ablation_synthetic()
        results["ablations"] = ablation_results
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (Synthetic Data)")
    print("=" * 60)
    print(f"{'Method':<25} {'BPW':>10} {'MSE':>12}")
    print("-" * 50)
    
    all_results = []
    for method, method_results in results.items():
        if method == "ablations":
            continue
        for r in method_results:
            all_results.append({
                "method": r.get("config", method),
                "bpw": r.get("bits_per_weight", 0),
                "mse": r.get("mse", 0),
            })
    
    # Sort by bits per weight
    all_results.sort(key=lambda x: x["bpw"])
    
    for r in all_results:
        print(f"{r['method']:<25} {r['bpw']:>10.3f} {r['mse']:>12.6f}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
