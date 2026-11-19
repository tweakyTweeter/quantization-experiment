"""
GPTQ baseline wrapper.

Wraps the auto-gptq library for comparison.
"""

import torch
from typing import Dict, Optional
import tempfile
import os


class GPTQWrapper:
    """
    Wrapper for GPTQ quantization using auto-gptq library.
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = True,
        device: str = "auto",
    ):
        """
        Initialize GPTQ wrapper.
        
        Args:
            bits: Bits per weight (2, 3, 4, 8).
            group_size: Quantization group size.
            desc_act: Use descending activation order.
            sym: Symmetric quantization.
            device: Target device.
        """
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.sym = sym
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def quantize_model(
        self,
        model_name_or_path: str,
        calibration_data: Optional[list] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Quantize a model using GPTQ.
        
        Args:
            model_name_or_path: HuggingFace model ID or path.
            calibration_data: Optional calibration data.
            output_dir: Output directory for quantized model.
            verbose: Print progress.
            
        Returns:
            stats: Quantization statistics.
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "auto-gptq not installed. Install with: pip install auto-gptq"
            )
        
        if verbose:
            print(f"GPTQ quantizing {model_name_or_path} to {self.bits} bits")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Prepare calibration data
        if calibration_data is None:
            from datasets import load_dataset
            
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            calibration_data = [
                tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                for text in dataset["text"][:128]
                if len(text) > 100
            ]
        
        # Configure quantization
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            sym=self.sym,
        )
        
        # Load and quantize model
        model = AutoGPTQForCausalLM.from_pretrained(
            model_name_or_path,
            quantize_config=quantize_config,
            trust_remote_code=True,
        )
        
        model.quantize(calibration_data)
        
        # Save if output_dir provided
        if output_dir:
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)
        
        # Compute stats
        stats = {
            "method": "gptq",
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
        Load a GPTQ quantized model.
        
        Args:
            quantized_dir: Directory with quantized model.
            device: Target device.
            
        Returns:
            model: Quantized model.
            tokenizer: Tokenizer.
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("auto-gptq not installed")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(quantized_dir, trust_remote_code=True)
        model = AutoGPTQForCausalLM.from_quantized(
            quantized_dir,
            device=device,
            trust_remote_code=True,
        )
        
        return model, tokenizer


def run_gptq_baseline(
    model_name: str,
    bits: int = 4,
    group_size: int = 128,
    device: str = "auto",
    verbose: bool = True,
) -> Dict:
    """
    Run GPTQ baseline and return stats.
    
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
    
    wrapper = GPTQWrapper(bits=bits, group_size=group_size, device=device)
    
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
                print(f"GPTQ failed: {e}")
            return {"method": "gptq", "bits": bits, "error": str(e)}
