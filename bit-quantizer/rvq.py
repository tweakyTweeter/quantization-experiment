"""
Activation-aware Residual Vector Quantization (RVQ) weight codec.

This is the math that actually wins (the binary BitLinear autoencoder was measured
to be a weak codec). Core ideas, all standard-but-strong:

  * Vector quantization captures the L2 geometry of weight sub-vectors -> better
    rate-distortion than scalar round-to-nearest at low bit-rates.
  * Residual stages (RVQ / additive quantization, the core of AQLM) refine the
    approximation: v ~= sum_m C_m[a_m].
  * Activation-aware weighting: clusters/centroids minimize output error
    sum_i diag(H)_i (v_i - vhat_i)^2 with H = X^T X from calibration (GPTQ/AWQ insight).

Storage per matrix: M index planes (log2(K) bits each) + per-column fp16 scales +
small fp16 codebooks (M*K*d, amortized).  bits/weight ~= M*log2(K)/d + overheads.
"""

import math
import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def col_normalize(W):
    s = W.abs().amax(dim=0).clamp(min=1e-8)
    return W / s, s


def vectorize(Wn, d):
    c, n = Wn.shape
    assert c % d == 0
    return Wn.t().reshape(n * (c // d), d)


def devectorize(V, c, n, d):
    return V.reshape(n, c).t().contiguous()


def weight_tiles(hdiag, c, n, d):
    """Per-dim activation weights tiled to the [N,d] vector layout."""
    if hdiag is None:
        return None
    w = hdiag.clamp(min=1e-8).reshape(c // d, d)  # [c/d, d]
    return w.repeat(n, 1)  # [N, d]  (matches vectorize: row = j*(c/d)+block)


def _assign(V, C, wv, chunk=200000):
    """Nearest centroid under (optionally weighted) squared distance.

    Weighted distance is expanded as
        ||v-C||^2_w = sum_d w(v^2) - 2 (w*v)@C^T + w@(C^2)^T
    so it uses matmuls only (no [N,K,d] broadcast) -> low memory, fast on MPS.
    """
    N = V.shape[0]
    out = torch.empty(N, dtype=torch.long, device=V.device)
    if wv is not None:
        C2 = C.pow(2)  # [K,d]
    for i in range(0, N, chunk):
        v = V[i : i + chunk]
        if wv is None:
            d2 = torch.cdist(v, C).pow(2)
        else:
            w = wv[i : i + chunk]                       # [b,d]
            t1 = (w * v * v).sum(1, keepdim=True)       # [b,1]
            t2 = (w * v) @ C.t()                        # [b,K]
            t3 = w @ C2.t()                             # [b,K]
            d2 = t1 - 2 * t2 + t3
        out[i : i + chunk] = d2.argmin(1)
    return out


def _update(V, a, K, wv):
    """Weighted centroid update; empty clusters reseeded to random points."""
    d = V.shape[1]
    C = torch.zeros(K, d, device=V.device)
    if wv is None:
        ones = torch.ones(V.shape[0], 1, device=V.device)
        num = torch.zeros(K, d, device=V.device).index_add_(0, a, V)
        cnt = torch.zeros(K, 1, device=V.device).index_add_(0, a, ones)
        C = num / cnt.clamp(min=1)
        empty = (cnt.squeeze(1) == 0)
    else:
        num = torch.zeros(K, d, device=V.device).index_add_(0, a, V * wv)
        den = torch.zeros(K, d, device=V.device).index_add_(0, a, wv)
        C = num / den.clamp(min=1e-8)
        empty = (den.sum(1) == 0)
    if empty.any():
        ridx = torch.randint(0, V.shape[0], (int(empty.sum()),), device=V.device)
        C[empty] = V[ridx]
    return C


def _kpp_init(V, K, wv, seed):
    """k-means++ seeding (greedy, weighted distance) for better/faster convergence."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = V.shape[0]
    first = torch.randint(0, N, (1,), generator=g).item()
    centers = [V[first]]
    C = torch.stack(centers)
    closest = _sqdist_to_set(V, C, wv)  # [N]
    for _ in range(1, K):
        probs = (closest / closest.sum().clamp(min=1e-12)).cpu()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(V[idx])
        newd = _sqdist_to_set(V, V[idx : idx + 1], wv)
        closest = torch.minimum(closest, newd)
    return torch.stack(centers)


def _sqdist_to_set(V, C, wv, chunk=200000):
    N = V.shape[0]
    out = torch.empty(N, device=V.device)
    C2 = C.pow(2) if wv is not None else None
    for i in range(0, N, chunk):
        v = V[i : i + chunk]
        if wv is None:
            d2 = torch.cdist(v, C).pow(2)
        else:
            w = wv[i : i + chunk]
            d2 = (w * v * v).sum(1, keepdim=True) - 2 * (w * v) @ C.t() + w @ C2.t()
        out[i : i + chunk] = d2.min(1).values.clamp(min=0)
    return out


def kmeans(V, K, wv=None, iters=15, seed=0, kpp=False):
    if kpp:
        C = _kpp_init(V, K, wv, seed)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        C = V[torch.randperm(V.shape[0], generator=g)[:K].to(V.device)].clone()
    a = _assign(V, C, wv)
    for _ in range(iters):
        C = _update(V, a, K, wv)
        a = _assign(V, C, wv)
    return C, a


def rvq_encode(V, M, K, wv=None, iters=12):
    """Residual VQ: returns list of codebooks and index planes."""
    codebooks, planes = [], []
    r = V.clone()
    for m in range(M):
        C, a = kmeans(r, K, wv=wv, iters=iters, seed=m)
        codebooks.append(C)
        planes.append(a)
        r = r - C[a]
    return codebooks, planes


def rvq_reconstruct(codebooks, planes):
    Vhat = None
    for C, a in zip(codebooks, planes):
        term = C[a]
        Vhat = term if Vhat is None else Vhat + term
    return Vhat


def bits_per_weight(d, M, K, c, n, scale_bits=16):
    code = M * math.log2(K) / d
    scale = scale_bits / c                      # per-column scale
    cb = M * K * d * scale_bits / (c * n)       # codebooks (fp16), amortized per matrix
    return code + scale + cb


def compress_matrix(W, d, M, K, hdiag=None, iters=12, device=DEV):
    c, n = W.shape
    Wn, s = col_normalize(W.float().to(device))
    V = vectorize(Wn, d)
    wv = weight_tiles(hdiag.to(device) if hdiag is not None else None, c, n, d)
    codebooks, planes = rvq_encode(V, M, K, wv=wv, iters=iters)
    Vhat = rvq_reconstruct(codebooks, planes)
    W_hat = devectorize(Vhat, c, n, d) * s
    bpw = bits_per_weight(d, M, K, c, n)
    return W_hat, bpw
