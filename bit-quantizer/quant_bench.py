"""
Quantization benchmark harness for GPT-2 on WikiText-2.

Every method produces a dequantized weight tensor W_hat per matrix; we load W_hat
into the FP32 model and measure WikiText-2 perplexity through one identical harness.
This isolates "quality at X bits/weight" and makes all methods directly comparable.

Targets: the 48 Conv1D weight matrices in the transformer blocks (68% of params).
Embeddings / LayerNorm / lm_head stay FP32 (standard practice).
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------------
# Model / data
# ----------------------------------------------------------------------
def load_model():
    return GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()


def load_tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


def get_targets(model):
    """Return dict name -> Conv1D module whose .weight we quantize."""
    return {n: m for n, m in model.named_modules() if isinstance(m, Conv1D)}


def total_quant_weights(model):
    return sum(m.weight.numel() for m in get_targets(model).values())


# ----------------------------------------------------------------------
# Perplexity (identical harness for every method)
# ----------------------------------------------------------------------
_DATA_CACHE = {}


def _get_ids(tok, split="test", max_tokens=None):
    key = (split, max_tokens)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids
    if max_tokens:
        ids = ids[:, :max_tokens]
    _DATA_CACHE[key] = ids
    return ids


@torch.no_grad()
def perplexity(model, tok, ctx=512, stride=512, max_tokens=40000, split="test"):
    ids = _get_ids(tok, split=split, max_tokens=max_tokens)
    nll = 0.0
    ntok = 0
    for i in range(0, ids.size(1) - 1, stride):
        a = ids[:, i : i + ctx].to(DEV)
        if a.size(1) < 2:
            break
        out = model(a, labels=a)
        n = a.size(1) - 1
        nll += out.loss.item() * n
        ntok += n
    return torch.exp(torch.tensor(nll / ntok)).item()


# ----------------------------------------------------------------------
# Apply a per-matrix transform W -> W_hat to all targets
# ----------------------------------------------------------------------
@torch.no_grad()
def snapshot_weights(model):
    return {n: m.weight.detach().clone() for n, m in get_targets(model).items()}


@torch.no_grad()
def restore_weights(model, snap):
    for n, m in get_targets(model).items():
        m.weight.copy_(snap[n])


@torch.no_grad()
def apply_transform(model, snap, fn, hess=None):
    """fn(name, W[, h]) -> W_hat ; writes W_hat into the model in-place."""
    for n, m in get_targets(model).items():
        W = snap[n].to(DEV)
        h = None if hess is None else hess[n]
        What = fn(n, W, h) if h is not None else fn(n, W)
        m.weight.copy_(What.to(m.weight.dtype))


# ----------------------------------------------------------------------
# Calibration: diagonal of the per-matrix Hessian  H = X^T X  (activation^2)
# This is the GPTQ/AWQ output-error weighting, diagonal approximation.
# ----------------------------------------------------------------------
@torch.no_grad()
def calibrate_diag_hessian(model, tok, n_seq=32, ctx=512):
    targets = get_targets(model)
    sums = {n: torch.zeros(m.weight.shape[0], device=DEV) for n, m in targets.items()}
    counts = {n: 0 for n in targets}
    handles = []

    def mk_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1])  # [tokens, in]
            sums[name] += (x * x).sum(0)
            counts[name] += x.shape[0]
        return hook

    for n, m in targets.items():
        handles.append(m.register_forward_hook(mk_hook(n)))

    ids = _get_ids(tok, split="train", max_tokens=ctx * n_seq + 10)
    for i in range(n_seq):
        a = ids[:, i * ctx : (i + 1) * ctx].to(DEV)
        if a.size(1) < 2:
            break
        model(a)

    for h in handles:
        h.remove()
    # diag(H) per input channel = mean activation^2 ; clamp for stability
    return {n: (sums[n] / max(counts[n], 1)).clamp(min=1e-8) for n in targets}


# ----------------------------------------------------------------------
# Bits-per-weight accounting helper
# ----------------------------------------------------------------------
def bpw_group_int(bits, group_size, scale_bits=16, zeropoint=False):
    """RTN/group integer quant: payload bits + amortized scale (+ zeropoint)."""
    overhead = scale_bits / group_size
    if zeropoint:
        overhead += scale_bits / group_size
    return bits + overhead


if __name__ == "__main__":
    tok = load_tokenizer()
    model = load_model()
    ppl = perplexity(model, tok)
    print(f"FP32 baseline ppl = {ppl:.3f}")
    print(f"quantizable weights = {total_quant_weights(model):,}")
