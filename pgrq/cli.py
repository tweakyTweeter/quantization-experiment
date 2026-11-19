"""
PGR-Q Command Line Interface.

Usage:
    pgrq quantize <model> --output <dir> [options]
    pgrq benchmark <model> --methods pgrq,gptq,awq,rtn [options]
    pgrq evaluate <quantized_dir>
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="PGR-Q: Post-training Quantization for LLM Weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Quantize command
    quant_parser = subparsers.add_parser("quantize", help="Quantize a model")
    quant_parser.add_argument("model", type=str, help="HuggingFace model name or path")
    quant_parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    quant_parser.add_argument("--bits", type=float, default=4.0, help="Target bits per weight")
    quant_parser.add_argument("--block-size", type=int, default=64, help="Block size (power of 2)")
    quant_parser.add_argument("--num-groups", type=int, default=4, help="Groups per block")
    quant_parser.add_argument("--base-codebook-size", type=int, default=256, help="K0")
    quant_parser.add_argument("--residual-levels", type=int, default=2, help="M residual levels")
    quant_parser.add_argument("--residual-codebook-size", type=int, default=256, help="K per level")
    quant_parser.add_argument("--no-joint-refinement", action="store_true", help="Disable joint refinement")
    quant_parser.add_argument("--phase-correction", action="store_true", help="Enable phase correction")
    quant_parser.add_argument("--scalar-floor", action="store_true", help="Enable scalar floor")
    quant_parser.add_argument("--no-entropy-coding", action="store_true", help="Use mixed-radix instead")
    quant_parser.add_argument("--calibration-samples", type=int, default=128, help="Calibration samples")
    quant_parser.add_argument("--calibration-dataset", type=str, default="wikitext", help="wikitext or c4")
    quant_parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    quant_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    quant_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark quantization methods")
    bench_parser.add_argument("model", type=str, help="HuggingFace model name or path")
    bench_parser.add_argument("--methods", type=str, default="pgrq,rtn",
                             help="Comma-separated methods: pgrq,gptq,awq,rtn,turbo")
    bench_parser.add_argument("--bits", type=str, default="4,3,2.5",
                             help="Comma-separated bit targets")
    bench_parser.add_argument("--output", "-o", type=str, help="Output file for results (JSON/CSV)")
    bench_parser.add_argument("--device", type=str, default="auto", help="Device")
    bench_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a quantized model")
    eval_parser.add_argument("quantized_dir", type=str, help="Directory with quantized model")
    eval_parser.add_argument("--dataset", type=str, default="wikitext", help="Evaluation dataset")
    eval_parser.add_argument("--device", type=str, default="auto", help="Device")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "quantize":
        run_quantize(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "evaluate":
        run_evaluate(args)


def run_quantize(args):
    """Run quantization command."""
    from pgrq.core.config import PGRQConfig, ScalarFloorType
    from pgrq.hf.integration import quantize_model
    
    # Build config
    config = PGRQConfig(
        block_size=args.block_size,
        num_groups=args.num_groups,
        base_codebook_size=args.base_codebook_size,
        num_residual_levels=args.residual_levels,
        residual_codebook_size=args.residual_codebook_size,
        joint_refinement=not args.no_joint_refinement,
        phase_correction=args.phase_correction,
        enable_scalar_floor=args.scalar_floor,
        entropy_coding=not args.no_entropy_coding,
        target_bits_per_weight=args.bits,
        calibration_samples=args.calibration_samples,
        calibration_dataset=args.calibration_dataset,
        device=args.device,
        seed=args.seed,
    )
    
    print(f"Quantizing {args.model} with PGR-Q")
    print(f"Configuration: {args.bits} bits target")
    
    stats = quantize_model(
        args.model,
        args.output,
        config=config,
        calibration_samples=args.calibration_samples,
        calibration_dataset=args.calibration_dataset,
        device=args.device,
        verbose=args.verbose,
    )
    
    print(f"\nQuantization complete!")
    print(f"Output saved to: {args.output}")
    print(f"Final bits per weight: {stats['bits_per_weight']:.3f}")


def run_benchmark(args):
    """Run benchmark command."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    methods = args.methods.split(",")
    bits_list = [float(b) for b in args.bits.split(",")]
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Benchmarking {args.model}")
    print(f"Methods: {methods}")
    print(f"Bit targets: {bits_list}")
    print(f"Device: {device}")
    print()
    
    results = []
    
    # Load baseline model for perplexity reference
    print("Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        baseline_model = baseline_model.to(device)
    
    # Evaluate baseline
    from pgrq.hf.evaluation import evaluate_perplexity
    
    print("Evaluating baseline perplexity...")
    baseline_ppl = evaluate_perplexity(
        baseline_model, tokenizer,
        dataset_name="wikitext",
        device=device,
        verbose=args.verbose,
    )
    
    results.append({
        "method": "baseline",
        "bits": 16.0,
        "perplexity": baseline_ppl["perplexity"],
    })
    
    print(f"Baseline perplexity: {baseline_ppl['perplexity']:.2f}")
    print()
    
    # Clean up baseline model
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Run each method
    for method in methods:
        for bits in bits_list:
            print(f"\n{'='*60}")
            print(f"Running {method} at {bits} bits...")
            print('='*60)
            
            try:
                result = run_single_benchmark(
                    args.model, method, bits, device, args.verbose
                )
                result["method"] = method
                result["bits"] = bits
                results.append(result)
                
                if "perplexity" in result:
                    print(f"Result: {method} {bits}b -> PPL {result['perplexity']:.2f}")
                    
            except Exception as e:
                print(f"Error running {method} at {bits} bits: {e}")
                results.append({
                    "method": method,
                    "bits": bits,
                    "error": str(e),
                })
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Method':<15} {'Bits':>8} {'Perplexity':>12}")
    print("-"*40)
    
    for r in results:
        ppl = r.get("perplexity", "N/A")
        if isinstance(ppl, float):
            ppl = f"{ppl:.2f}"
        print(f"{r['method']:<15} {r['bits']:>8.1f} {ppl:>12}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        
        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        elif output_path.suffix == ".csv":
            import csv
            with open(output_path, "w", newline="") as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        
        print(f"\nResults saved to: {args.output}")
    
    return results


def run_single_benchmark(model_name, method, bits, device, verbose):
    """Run a single benchmark configuration."""
    import torch
    import tempfile
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pgrq.hf.evaluation import evaluate_perplexity
    
    result = {}
    
    if method == "pgrq":
        from pgrq.core.config import PGRQConfig
        from pgrq.hf.integration import quantize_model, load_quantized_model
        
        config = PGRQConfig(
            target_bits_per_weight=bits,
            device=device,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = quantize_model(
                model_name, tmpdir,
                config=config,
                device=device,
                verbose=verbose,
            )
            
            model, tokenizer = load_quantized_model(tmpdir, device=device, verbose=verbose)
            
            ppl = evaluate_perplexity(model, tokenizer, device=device, verbose=verbose)
            
            result["perplexity"] = ppl["perplexity"]
            result["bits_actual"] = stats["bits_per_weight"]
            
    elif method == "rtn":
        from pgrq.baselines.rtn import quantize_model_rtn
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to(device)
        
        int_bits = max(2, min(8, int(bits)))
        stats = quantize_model_rtn(model, bits=int_bits, device=device, verbose=verbose)
        
        ppl = evaluate_perplexity(model, tokenizer, device=device, verbose=verbose)
        
        result["perplexity"] = ppl["perplexity"]
        result["bits_actual"] = stats["bits_per_weight"]
        
    elif method == "turbo":
        from pgrq.baselines.turbo_scalar import quantize_model_turbo_scalar
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to(device)
        
        int_bits = max(1, min(4, int(bits)))
        stats = quantize_model_turbo_scalar(model, bits=int_bits, device=device, verbose=verbose)
        
        ppl = evaluate_perplexity(model, tokenizer, device=device, verbose=verbose)
        
        result["perplexity"] = ppl["perplexity"]
        result["bits_actual"] = stats["bits_per_weight"]
        
    elif method == "gptq":
        from pgrq.baselines.gptq_wrapper import run_gptq_baseline
        
        int_bits = max(2, min(8, int(bits)))
        result = run_gptq_baseline(model_name, bits=int_bits, device=device, verbose=verbose)
        
    elif method == "awq":
        from pgrq.baselines.awq_wrapper import run_awq_baseline
        
        result = run_awq_baseline(model_name, bits=4, device=device, verbose=verbose)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


def run_evaluate(args):
    """Run evaluation command."""
    from pgrq.hf.integration import load_quantized_model
    from pgrq.hf.evaluation import evaluate_model_quality
    
    print(f"Loading quantized model from {args.quantized_dir}")
    
    model, tokenizer = load_quantized_model(
        args.quantized_dir,
        device=args.device,
        verbose=args.verbose,
    )
    
    print("Evaluating model quality...")
    results = evaluate_model_quality(
        model, tokenizer,
        device=args.device,
        verbose=args.verbose,
    )
    
    print("\nEvaluation Results:")
    print(f"  WikiText-2 Perplexity: {results['wikitext2_ppl']:.2f}")
    print(f"  C4 Perplexity: {results['c4_ppl']:.2f}")


if __name__ == "__main__":
    main()
