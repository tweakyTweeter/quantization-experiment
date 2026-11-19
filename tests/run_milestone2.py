#!/usr/bin/env python
"""
Milestone 2: Full Pareto benchmark + ablations on OPT-125M with perplexity.

Runs:
1. Baseline (fp16) perplexity
2. PGR-Q at multiple bit configurations
3. RTN at 4/3/2 bits
4. TurboScalar at 4/3/2 bits
5. Ablation studies A-E
6. Generates results table and Pareto data
"""

import torch
import time
import json
import sys
import os
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

# Ensure we can import pgrq
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgrq.core.config import PGRQConfig
from pgrq.core.quantizer import PGRQuantizer
from pgrq.baselines.rtn import RTNQuantizer
from pgrq.baselines.turbo_scalar import TurboScalarQuantizer


MODEL_NAME = "facebook/opt-125m"
DEVICE = "cpu"  # Use CPU for OPT-125M (fits in RAM)
RESULTS_FILE = Path(__file__).parent.parent / "benchmark_results.json"


def load_model_and_tokenizer():
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model, tokenizer


def evaluate_perplexity_fast(model, tokenizer, max_tokens=None):
    """
    Correct perplexity evaluation on WikiText-2 test set.
    
    Uses sliding window with proper token counting.
    """
    from datasets import load_dataset
    import math
    
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    
    # Concatenate text
    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n\n".join(texts)
    
    # Tokenize
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    
    total_length = input_ids.shape[1]
    if max_tokens and total_length > max_tokens:
        input_ids = input_ids[:, :max_tokens]
        total_length = max_tokens
    
    seq_len = 2048
    stride = 1024
    
    nlls = []
    num_tokens_evaluated = 0
    
    model.eval()
    with torch.no_grad():
        for begin in range(0, total_length - 1, stride):
            end = min(begin + seq_len, total_length)
            
            input_chunk = input_ids[:, begin:end]
            target = input_chunk.clone()
            
            # Only count new tokens (avoid double-counting in overlap)
            if begin > 0:
                target[:, :-(end - begin - stride)] = -100
            
            outputs = model(input_chunk, labels=target)
            
            # Count non-masked tokens
            valid_tokens = (target != -100).sum().item() - 1  # -1 for shift
            if valid_tokens <= 0:
                valid_tokens = input_chunk.shape[1] - 1
            
            nlls.append(outputs.loss.item() * valid_tokens)
            num_tokens_evaluated += valid_tokens
            
            if end >= total_length:
                break
            
            # Limit evaluation for speed (first 10 chunks ~ 10K tokens)
            if len(nlls) >= 10:
                break
    
    ppl = math.exp(sum(nlls) / num_tokens_evaluated)
    
    return ppl, num_tokens_evaluated


def get_linear_weights(model):
    """Extract linear layer weights for quantization."""
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "embed" not in name.lower() and "lm_head" not in name.lower():
                weights[name] = module.weight.data.clone()
    return weights


def replace_weights(model, dequantized_weights):
    """Replace model weights with dequantized versions."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in dequantized_weights:
            module.weight.data = dequantized_weights[name].to(module.weight.dtype)


def run_pgrq_benchmark(model, tokenizer, weights, config_name, config):
    """Run PGR-Q quantization and evaluate perplexity."""
    print(f"\n  Running PGR-Q ({config_name})...")
    
    start_time = time.time()
    
    # Train codebooks
    quantizer = PGRQuantizer(config)
    weight_list = list(weights.values())
    # Use subset for training
    train_subset = weight_list[:min(20, len(weight_list))]
    quantizer.train_codebooks(train_subset, verbose=False)
    
    # Quantize all weights
    dequantized = {}
    total_compressed = 0
    total_elements = 0
    
    for name, w in weights.items():
        qt = quantizer.quantize(w, verbose=False)
        w_rec = quantizer.dequantize(qt)
        dequantized[name] = w_rec
        
        stats = quantizer.get_compression_stats(qt)
        total_compressed += stats["total_bytes"]
        total_elements += w.numel()
    
    quant_time = time.time() - start_time
    bits_per_weight = total_compressed * 8 / total_elements
    
    # Replace weights and evaluate
    original_weights = {name: model.state_dict()[name + ".weight"].clone() 
                       for name in dequantized.keys()
                       if name + ".weight" in model.state_dict()}
    
    replace_weights(model, dequantized)
    
    ppl, num_tokens = evaluate_perplexity_fast(model, tokenizer)
    
    # Restore original weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in original_weights:
            module.weight.data = original_weights[name]
    
    print(f"    BPW: {bits_per_weight:.3f}, PPL: {ppl:.2f}, Time: {quant_time:.1f}s")
    
    return {
        "method": f"PGR-Q ({config_name})",
        "bits_per_weight": bits_per_weight,
        "perplexity": ppl,
        "time_s": quant_time,
        "num_tokens": num_tokens,
    }


def run_rtn_benchmark(model, tokenizer, weights, bits):
    """Run RTN quantization and evaluate perplexity."""
    print(f"\n  Running RTN {bits}-bit...")
    
    start_time = time.time()
    
    quantizer = RTNQuantizer(bits=bits, group_size=128, device=DEVICE)
    
    dequantized = {}
    total_compressed = 0
    total_elements = 0
    
    for name, w in weights.items():
        qt, scales, zeros = quantizer.quantize(w)
        w_rec = quantizer.dequantize(qt, scales, zeros, w.shape)
        dequantized[name] = w_rec
        
        sizes = quantizer.get_compressed_size(w.shape)
        total_compressed += sizes["total_bytes"]
        total_elements += w.numel()
    
    quant_time = time.time() - start_time
    bits_per_weight = total_compressed * 8 / total_elements
    
    # Replace weights and evaluate
    original_weights = {name: model.state_dict()[name + ".weight"].clone()
                       for name in dequantized.keys()
                       if name + ".weight" in model.state_dict()}
    
    replace_weights(model, dequantized)
    
    ppl, num_tokens = evaluate_perplexity_fast(model, tokenizer)
    
    # Restore
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in original_weights:
            module.weight.data = original_weights[name]
    
    print(f"    BPW: {bits_per_weight:.3f}, PPL: {ppl:.2f}, Time: {quant_time:.1f}s")
    
    return {
        "method": f"RTN-{bits}b",
        "bits_per_weight": bits_per_weight,
        "perplexity": ppl,
        "time_s": quant_time,
        "num_tokens": num_tokens,
    }


def run_turbo_benchmark(model, tokenizer, weights, bits):
    """Run TurboScalar quantization and evaluate perplexity."""
    print(f"\n  Running TurboScalar {bits}-bit...")
    
    start_time = time.time()
    
    quantizer = TurboScalarQuantizer(bits=bits, block_size=64, device=DEVICE)
    
    dequantized = {}
    total_compressed = 0
    total_elements = 0
    
    for name, w in weights.items():
        codes, norms, shape_info = quantizer.quantize(w)
        w_rec = quantizer.dequantize(codes, norms, shape_info)
        dequantized[name] = w_rec
        
        sizes = quantizer.get_compressed_size(w.shape)
        total_compressed += sizes["total_bytes"]
        total_elements += w.numel()
    
    quant_time = time.time() - start_time
    bits_per_weight = total_compressed * 8 / total_elements
    
    # Replace weights and evaluate
    original_weights = {name: model.state_dict()[name + ".weight"].clone()
                       for name in dequantized.keys()
                       if name + ".weight" in model.state_dict()}
    
    replace_weights(model, dequantized)
    
    ppl, num_tokens = evaluate_perplexity_fast(model, tokenizer)
    
    # Restore
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in original_weights:
            module.weight.data = original_weights[name]
    
    print(f"    BPW: {bits_per_weight:.3f}, PPL: {ppl:.2f}, Time: {quant_time:.1f}s")
    
    return {
        "method": f"TurboScalar-{bits}b",
        "bits_per_weight": bits_per_weight,
        "perplexity": ppl,
        "time_s": quant_time,
        "num_tokens": num_tokens,
    }


def run_ablation(model, tokenizer, weights, ablation_name, config_a, config_b, label_a, label_b):
    """Run a single ablation comparison."""
    print(f"\n  Ablation [{ablation_name}]: {label_a} vs {label_b}")
    
    result_a = run_pgrq_benchmark(model, tokenizer, weights, label_a, config_a)
    result_b = run_pgrq_benchmark(model, tokenizer, weights, label_b, config_b)
    
    return {
        "ablation": ablation_name,
        label_a: result_a,
        label_b: result_b,
        "winner": label_a if result_a["perplexity"] < result_b["perplexity"] else label_b,
    }


def main():
    print("=" * 70)
    print("MILESTONE 2: Full Pareto Benchmark on OPT-125M")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Get weights
    weights = get_linear_weights(model)
    total_params = sum(w.numel() for w in weights.values())
    print(f"Quantizing {len(weights)} layers ({total_params:,} parameters)")
    
    results = []
    ablation_results = []
    
    # =====================================================
    # 1. BASELINE PERPLEXITY
    # =====================================================
    print("\n" + "=" * 70)
    print("1. BASELINE (fp32)")
    print("=" * 70)
    
    ppl_baseline, num_tokens = evaluate_perplexity_fast(model, tokenizer)
    print(f"  Baseline PPL: {ppl_baseline:.2f} ({num_tokens} tokens)")
    
    results.append({
        "method": "Baseline (fp32)",
        "bits_per_weight": 32.0,
        "perplexity": ppl_baseline,
        "time_s": 0,
        "num_tokens": num_tokens,
    })
    
    # =====================================================
    # 2. PGR-Q at multiple configs
    # =====================================================
    print("\n" + "=" * 70)
    print("2. PGR-Q QUANTIZATION")
    print("=" * 70)
    
    pgrq_configs = [
        ("low-bit", PGRQConfig(
            block_size=64, num_groups=4,
            base_codebook_size=64,
            num_residual_levels=1,
            residual_codebook_size=32,
            kmeans_iters=50, joint_refinement=True,
            entropy_coding=True,
        )),
        ("medium", PGRQConfig(
            block_size=64, num_groups=4,
            base_codebook_size=128,
            num_residual_levels=2,
            residual_codebook_size=64,
            kmeans_iters=50, joint_refinement=True,
            entropy_coding=True,
        )),
        ("high", PGRQConfig(
            block_size=64, num_groups=4,
            base_codebook_size=256,
            num_residual_levels=2,
            residual_codebook_size=256,
            kmeans_iters=50, joint_refinement=True,
            entropy_coding=True,
        )),
    ]
    
    for name, config in pgrq_configs:
        result = run_pgrq_benchmark(model, tokenizer, weights, name, config)
        results.append(result)
    
    # =====================================================
    # 3. RTN BASELINES
    # =====================================================
    print("\n" + "=" * 70)
    print("3. RTN BASELINES")
    print("=" * 70)
    
    for bits in [4, 3, 2]:
        result = run_rtn_benchmark(model, tokenizer, weights, bits)
        results.append(result)
    
    # =====================================================
    # 4. TURBOSCALAR BASELINES
    # =====================================================
    print("\n" + "=" * 70)
    print("4. TURBOSCALAR BASELINES")
    print("=" * 70)
    
    for bits in [4, 3, 2]:
        result = run_turbo_benchmark(model, tokenizer, weights, bits)
        results.append(result)
    
    # =====================================================
    # 5. ABLATION STUDIES
    # =====================================================
    print("\n" + "=" * 70)
    print("5. ABLATION STUDIES")
    print("=" * 70)
    
    base_config = PGRQConfig(
        block_size=64, num_groups=4,
        base_codebook_size=128,
        num_residual_levels=2,
        residual_codebook_size=64,
        kmeans_iters=50,
        entropy_coding=True,
    )
    
    # A. Phase correction ON vs OFF
    config_a_off = deepcopy(base_config)
    config_a_off.phase_correction = False
    config_a_on = deepcopy(base_config)
    config_a_on.phase_correction = True
    
    abl_a = run_ablation(model, tokenizer, weights,
                         "A: Phase", config_a_off, config_a_on,
                         "Phase OFF", "Phase ON")
    ablation_results.append(abl_a)
    
    # B. Beta-scalar floor vs digit-plane
    config_b_beta = deepcopy(base_config)
    config_b_beta.enable_scalar_floor = True
    config_b_beta.scalar_floor_type = PGRQConfig().scalar_floor_type  # BETA_MAX_LLOYD
    
    config_b_digit = deepcopy(base_config)
    config_b_digit.enable_scalar_floor = True
    from pgrq.core.config import ScalarFloorType
    config_b_digit.scalar_floor_type = ScalarFloorType.DIGIT_PLANES
    
    abl_b = run_ablation(model, tokenizer, weights,
                         "B: Scalar floor", config_b_beta, config_b_digit,
                         "Beta-Max-Lloyd", "Digit-planes")
    ablation_results.append(abl_b)
    
    # C. Greedy RVQ vs Joint-refined
    config_c_greedy = deepcopy(base_config)
    config_c_greedy.joint_refinement = False
    config_c_joint = deepcopy(base_config)
    config_c_joint.joint_refinement = True
    
    abl_c = run_ablation(model, tokenizer, weights,
                         "C: RVQ refinement", config_c_greedy, config_c_joint,
                         "Greedy", "Joint-refined")
    ablation_results.append(abl_c)
    
    # D. Activation-aware vs raw-magnitude importance
    # (For this we need calibration data - skip for now, use uniform allocation)
    print("\n  Ablation [D: Importance]: Skipped (requires calibration forward pass)")
    ablation_results.append({
        "ablation": "D: Importance",
        "note": "Requires calibration forward pass - both configs use uniform allocation in this test",
    })
    
    # E. Per-layer vs cross-layer shared codebooks
    config_e_shared = deepcopy(base_config)
    config_e_shared.cross_layer_codebook_sharing = True
    config_e_perlayer = deepcopy(base_config)
    config_e_perlayer.cross_layer_codebook_sharing = False
    
    abl_e = run_ablation(model, tokenizer, weights,
                         "E: Codebook sharing", config_e_shared, config_e_perlayer,
                         "Cross-layer", "Per-layer")
    ablation_results.append(abl_e)
    
    # =====================================================
    # FINAL REPORT
    # =====================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # Sort by bits_per_weight
    results_sorted = sorted(results, key=lambda x: x["bits_per_weight"])
    
    print(f"\n{'Method':<25} {'BPW':>8} {'Perplexity':>12} {'Time':>8}")
    print("-" * 58)
    for r in results_sorted:
        ppl_str = f"{r['perplexity']:.2f}"
        time_str = f"{r['time_s']:.1f}s" if r['time_s'] > 0 else "-"
        print(f"{r['method']:<25} {r['bits_per_weight']:>8.3f} {ppl_str:>12} {time_str:>8}")
    
    # Ablation summary
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")
    
    for abl in ablation_results:
        if "note" in abl:
            print(f"\n[{abl['ablation']}]: {abl['note']}")
            continue
        
        a_key = [k for k in abl.keys() if k not in ("ablation", "winner")][0]
        b_key = [k for k in abl.keys() if k not in ("ablation", "winner", a_key)][0]
        
        ppl_a = abl[a_key]["perplexity"]
        ppl_b = abl[b_key]["perplexity"]
        bpw_a = abl[a_key]["bits_per_weight"]
        bpw_b = abl[b_key]["bits_per_weight"]
        
        print(f"\n[{abl['ablation']}]")
        print(f"  {a_key}: BPW={bpw_a:.3f}, PPL={ppl_a:.2f}")
        print(f"  {b_key}: BPW={bpw_b:.3f}, PPL={ppl_b:.2f}")
        print(f"  Winner: {abl['winner']}")
    
    # Save results
    all_results = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "pareto": results,
        "ablations": ablation_results,
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    # Generate CSV
    csv_file = RESULTS_FILE.with_suffix(".csv")
    with open(csv_file, "w") as f:
        f.write("Method,Bits_Per_Weight,Perplexity,Time_s\n")
        for r in results_sorted:
            f.write(f"{r['method']},{r['bits_per_weight']:.3f},{r['perplexity']:.2f},{r['time_s']:.1f}\n")
    
    print(f"CSV saved to: {csv_file}")
    
    return all_results


if __name__ == "__main__":
    main()
