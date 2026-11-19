"""
AWQ baseline wrapper.

Wraps the AutoAWQ library for comparison.
"""

import torch
from typing import Dict, Optional
import tempfile


class AWQWrapper:
    """
    Wrapper for AWQ quantization using AutoAWQ library.
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        device: str = "auto",
    ):
        """
        Initialize AWQ wrapper.
        
        Args:
            bits: Bits per weight (typically 4).
            group_size: Quantization group size.
            zero_point: Use zero-point quantization.
            device: Target device.
        """
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def quantize_model(
        self,
        model_name_or_path: str,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Quantize a model using AWQ.
        
        Args:
            model_name_or_path: HuggingFace model ID or path.
            output_dir: Output directory for quantized model.
            verbose: Print progress.
            
        Returns:
            stats: Quantization statistics.
        """
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "autoawq not installed. Install with: pip install autoawq"
            )
        
        if verbose:
            print(f"AWQ quantizing {model_name_or_path} to {self.bits} bits")
        
        # Load model
        model = AutoAWQForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Configure quantization
        quant_config = {
            "zero_point": self.zero_point,
            "q_group_size": self.group_size,
            "w_bit": self.bits,
            "version": "GEMM",
        }
        
        # Quantize
        model.quantize(tokenizer, quant_config=quant_config)
        
        # Save if output_dir provided
        if output_dir:
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)
        
        stats = {
            "method": "awq",
            "bits": self.bits,
            "group_size": self.group_size,
        }
        
        return stats, model, tokenizer
    
    @staticmethod
    def load_quantized(
        quantized_dir: str,
        device: str = "auto",
    ):
        """
        Load an AWQ quantized model.
        
        Args:
            quantized_dir: Directory with quantized model.
            device: Target device.
            
        Returns:
            model: Quantized model.
            tokenizer: Tokenizer.
        """
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("autoawq not installed")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoAWQForCausalLM.from_quantized(
            quantized_dir,
            fuse_layers=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_dir, trust_remote_code=True)
        
        return model, tokenizer


def run_awq_baseline(
    model_name: str,
    bits: int = 4,
    group_size: int = 128,
    device: str = "auto",
    verbose: bool = True,
) -> Dict:
    """
    Run AWQ baseline and return stats.
    
    Args:
        model_name: HuggingFace model name.
        bits: Bits per weight.
        group_size: Group size.
        device: Target device.
        verbose: Print progress.
        
    Returns:
        results: Dict with stats and perplexity.
    """
    from pgrq.hf.evaluation import evaluate_perplexity
    
    wrapper = AWQWrapper(bits=bits, group_size=group_size, device=device)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            stats, model, tokenizer = wrapper.quantize_model(
                model_name,
                output_dir=tmpdir,
                verbose=verbose,
            )
            
            # Evaluate perplexity
            ppl_results = evaluate_perplexity(
                model.model if hasattr(model, 'model') else model,
                tokenizer,
                dataset_name="wikitext",
                device=device,
                verbose=verbose,
            )
            
            stats["perplexity"] = ppl_results["perplexity"]
            
            return stats
            
        except Exception as e:
            if verbose:
                print(f"AWQ failed: {e}")
            return {"method": "awq", "bits": bits, "error": str(e)}
