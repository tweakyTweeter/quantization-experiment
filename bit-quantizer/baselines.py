"""
Baseline curve: HF optimum-quanto group-wise integer quantization (RTN-style).

NOTE on the named SOTA methods:
  - AWQ (autoawq), GPTQ (auto-gptq/gptqmodel), AQLM (aqlm) ship CUDA-only kernels
    and do not run on this Mac (MPS/CPU). quanto is the strongest *runnable* real
    package here; it is round-to-nearest with group scales (similar in spirit to
    GGUF's non-K Q-types and to bitsandbytes NF/INT). GGUF k-quants / GPTQ / AQLM
    are stronger than plain RTN, so beating quanto is necessary but not sufficient
    to claim the published SOTA crown -- reported honestly in the final summary.
"""

import torch
import optimum.quanto as q
from optimum.quanto import quantize_weight, qint2, qint4, qint8, MaxOptimizer
import quant_bench as qb

_OPT = MaxOptimizer()
_QT = {2: qint2, 4: qint4, 8: qint8}


def quanto_dequant(W, bits, group_size, axis=0, zeropoint=False):
    qt = _QT[bits]
    scale, shift = _OPT(W, qtype=qt, axis=axis, group_size=group_size, zeropoint=zeropoint)
    qw = quantize_weight(W, qtype=qt, axis=axis, scale=scale, shift=shift, group_size=group_size)
    return qw.dequantize()


def run_quanto_curve(model, tok, snap, configs):
    rows = []
    for bits, gs, zp in configs:
        fn = lambda n, W, b=bits, g=gs, z=zp: quanto_dequant(W, b, g, axis=0, zeropoint=z)
        qb.apply_transform(model, snap, fn)
        ppl = qb.perplexity(model, tok)
        bpw = qb.bpw_group_int(bits, gs, zeropoint=zp)
        rows.append(("quanto", f"int{bits}_gs{gs}{'_zp' if zp else ''}", bpw, ppl))
        print(f"  quanto int{bits} gs{gs}{' zp' if zp else ''}: bpw={bpw:.3f}  ppl={ppl:.3f}")
        qb.restore_weights(model, snap)
    return rows


if __name__ == "__main__":
    tok = qb.load_tokenizer()
    model = qb.load_model()
    snap = qb.snapshot_weights(model)
    fp = qb.perplexity(model, tok)
    print(f"FP32 baseline ppl = {fp:.3f}\n")
    configs = [
        (4, 128, False),
        (4, 64, False),
        (4, 32, False),
        (2, 64, True),
        (2, 32, True),
        (2, 16, True),
    ]
    rows = run_quanto_curve(model, tok, snap, configs)
    print("\nmethod, config, bpw, ppl")
    for r in rows:
        print(r)
