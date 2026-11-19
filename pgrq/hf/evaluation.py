"""
Perplexity evaluation for quantized models.

Evaluates perplexity on WikiText-2 and C4 test sets.
"""

import torch
from typing import Optional, Dict, List
from tqdm import tqdm
import math


def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    split: str = "test",
    max_samples: Optional[int] = None,
    seq_length: int = 2048,
    stride: int = 512,
    device: str = "auto",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate perplexity on a dataset.
    
    Uses sliding window approach for long sequences.
    
    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        dataset_name: "wikitext" or "c4".
        split: Dataset split ("test", "validation").
        max_samples: Maximum samples to evaluate (None for all).
        seq_length: Context window size.
        stride: Stride for sliding window.
        device: Target device.
        verbose: Print progress.
        
    Returns:
        results: Dict with perplexity and other metrics.
    """
    from datasets import load_dataset
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_column = "text"
    elif dataset_name == "c4":
        # C4 test split
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Concatenate all text
    if dataset_name == "c4":
        texts = []
        for i, example in enumerate(dataset):
            texts.append(example[text_column])
            if max_samples and i >= max_samples:
                break
        full_text = "\n\n".join(texts)
    else:
        texts = [ex[text_column] for ex in dataset if ex[text_column].strip()]
        if max_samples:
            texts = texts[:max_samples]
        full_text = "\n\n".join(texts)
    
    # Tokenize
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids
    
    if verbose:
        print(f"Evaluating on {input_ids.shape[1]} tokens")
    
    # Evaluate using sliding window
    model.eval()
    
    nlls = []
    prev_end_loc = 0
    
    num_windows = (input_ids.shape[1] - 1) // stride
    
    with torch.no_grad():
        for begin_loc in tqdm(range(0, input_ids.shape[1] - 1, stride), 
                             desc="Evaluating", disable=not verbose):
            end_loc = min(begin_loc + seq_length, input_ids.shape[1])
            trg_len = end_loc - prev_end_loc  # May be different from stride on last loop
            
            input_chunk = input_ids[:, begin_loc:end_loc].to(device)
            target_chunk = input_chunk.clone()
            target_chunk[:, :-trg_len] = -100  # Mask already-computed tokens
            
            try:
                outputs = model(input_chunk, labels=target_chunk)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood.item())
            except Exception as e:
                if verbose:
                    print(f"Warning: Evaluation failed at position {begin_loc}: {e}")
                continue
            
            prev_end_loc = end_loc
            
            if end_loc >= input_ids.shape[1] - 1:
                break
    
    # Compute perplexity
    total_nll = sum(nlls)
    num_tokens = prev_end_loc
    
    ppl = math.exp(total_nll / num_tokens)
    
    results = {
        "perplexity": ppl,
        "total_nll": total_nll,
        "num_tokens": num_tokens,
        "dataset": dataset_name,
        "split": split,
    }
    
    if verbose:
        print(f"Perplexity on {dataset_name} ({split}): {ppl:.2f}")
    
    return results


def evaluate_model_quality(
    model,
    tokenizer,
    device: str = "auto",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model quality on standard benchmarks.
    
    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        device: Target device.
        verbose: Print progress.
        
    Returns:
        results: Dict with all metrics.
    """
    results = {}
    
    # WikiText-2 perplexity
    if verbose:
        print("\nEvaluating WikiText-2 perplexity...")
    wt2_results = evaluate_perplexity(
        model, tokenizer,
        dataset_name="wikitext",
        split="test",
        device=device,
        verbose=verbose,
    )
    results["wikitext2_ppl"] = wt2_results["perplexity"]
    
    # C4 perplexity (subset for speed)
    if verbose:
        print("\nEvaluating C4 perplexity...")
    c4_results = evaluate_perplexity(
        model, tokenizer,
        dataset_name="c4",
        split="validation",
        max_samples=256,
        device=device,
        verbose=verbose,
    )
    results["c4_ppl"] = c4_results["perplexity"]
    
    return results


def compute_reconstruction_error(
    original_model,
    quantized_model,
    device: str = "auto",
) -> Dict[str, float]:
    """
    Compute reconstruction error between original and quantized weights.
    
    Args:
        original_model: Original HuggingFace model.
        quantized_model: Quantized model.
        device: Target device.
        
    Returns:
        errors: Dict with error metrics per layer and overall.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    errors = {"layers": {}}
    total_mse = 0.0
    total_elements = 0
    
    orig_state = original_model.state_dict()
    quant_state = quantized_model.state_dict()
    
    for name in orig_state.keys():
        if name in quant_state:
            orig_w = orig_state[name].to(device).float()
            quant_w = quant_state[name].to(device).float()
            
            if orig_w.shape == quant_w.shape:
                mse = ((orig_w - quant_w) ** 2).mean().item()
                rel_error = mse / (orig_w.pow(2).mean().item() + 1e-10)
                
                errors["layers"][name] = {
                    "mse": mse,
                    "relative_error": rel_error,
                    "numel": orig_w.numel(),
                }
                
                total_mse += mse * orig_w.numel()
                total_elements += orig_w.numel()
    
    if total_elements > 0:
        errors["overall_mse"] = total_mse / total_elements
    else:
        errors["overall_mse"] = 0.0
    
    return errors
