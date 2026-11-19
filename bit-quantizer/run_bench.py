"""
End-to-end benchmark: RVQ codec vs quanto RTN on GPT-2 / WikiText-2, identical harness.
Reports perplexity vs average bits-per-weight (params-weighted) for every method.
"""

import time
import torch
import quant_bench as qb
import rvq
import baselines as bl


def avg_bpw_rvq(model, d, M, K):
    tot_bits = 0.0
    tot_w = 0
    for n, mod in qb.get_targets(model).items():
        c, nn_ = mod.weight.shape
        tot_bits += rvq.bits_per_weight(d, M, K, c, nn_) * (c * nn_)
        tot_w += c * nn_
    return tot_bits / tot_w


def run_rvq(model, tok, snap, hess, d, M, K, use_hess=True, iters=12):
    def fn(name, W):
        h = hess[name] if use_hess else None
        What, _ = rvq.compress_matrix(W, d, M, K, hdiag=h, iters=iters)
        return What
    t = time.time()
    qb.apply_transform(model, snap, fn)
    ppl = qb.perplexity(model, tok)
    qb.restore_weights(model, snap)
    bpw = avg_bpw_rvq(model, d, M, K)
    tag = f"RVQ d{d} M{M} K{K}{'+H' if use_hess else ''}"
    print(f"  {tag:22s} bpw={bpw:.3f}  ppl={ppl:.3f}  ({time.time()-t:.0f}s)")
    return (tag, bpw, ppl)


if __name__ == "__main__":
    tok = qb.load_tokenizer()
    model = qb.load_model()
    snap = qb.snapshot_weights(model)
    fp = qb.perplexity(model, tok)
    print(f"FP32 baseline ppl = {fp:.3f}")
    print("calibrating diag-Hessian ...")
    hess = qb.calibrate_diag_hessian(model, tok, n_seq=24)

    print("\n-- quanto RTN baselines --")
    for bits, gs, zp in [(4, 64, False), (4, 32, False), (2, 16, True), (2, 32, True)]:
        fn = lambda n, W, b=bits, g=gs, z=zp: bl.quanto_dequant(W, b, g, zeropoint=z)
        qb.apply_transform(model, snap, fn)
        ppl = qb.perplexity(model, tok)
        qb.restore_weights(model, snap)
        print(f"  quanto int{bits} gs{gs}: bpw={qb.bpw_group_int(bits, gs, zeropoint=zp):.3f}  ppl={ppl:.3f}")

    print("\n-- RVQ (activation-aware) --")
    results = []
    for d, M, K in [(4, 1, 256), (8, 2, 256), (8, 3, 256), (4, 2, 256), (8, 4, 256)]:
        results.append(run_rvq(model, tok, snap, hess, d, M, K, use_hess=True))

    print("\n-- RVQ (no activation weighting, ablation) --")
    for d, M, K in [(8, 3, 256), (8, 4, 256)]:
        results.append(run_rvq(model, tok, snap, hess, d, M, K, use_hess=False))

    print(f"\nFP32 ppl={fp:.3f}")
    print("method, bpw, ppl")
    for r in sorted(results, key=lambda x: x[1]):
        print(r)
