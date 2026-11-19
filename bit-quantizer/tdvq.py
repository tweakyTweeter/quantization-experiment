"""
TDVQ: Ternary-Decoder Vector Quantization.

A learned weight codec built from BitNet BitLinear (ternary {-1,0,1}) layers, used to
compress LLM weight matrices and decompress them at load time.

Pipeline per weight matrix W [in=c, out=n]:
  1. Per-output-column normalization: s_j = absmax(W[:, j]); Wn = W / s   (s stored fp16)
  2. Vectorize along the input dim into d-dim vectors V  [N, d], N = c*n/d
  3. Encode each vector to a B-bit binary code with the shared ternary encoder
     (sigmoid + hard-threshold straight-through), optionally refined by activation-aware
     bit-flip search.  -> stored payload = B bits / vector
  4. Decode with the shared ternary decoder D: {0,1}^B -> R^d, rescale by s.

Storage = per-matrix codes (B bits/vector) + per-column scales (fp16) + ONE shared
ternary codec (amortized over the whole model -> ~0 bits/weight).
bits/weight ~= B/d + 16/c.

Only the SHARED codec carries the ternary constraint; it is verified by
verify_ternary_constraint().  The per-matrix payload is pure bits (the codes).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- BitLinear
class BitLinear(nn.Module):
    """Ternary {-1,0,1} * scale weights with straight-through estimator (BitNet)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def ternarize(self, w):
        scale = w.abs().mean().clamp(min=1e-5)
        w_tern = torch.round(w / scale).clamp(-1, 1)
        return w + (w_tern * scale - w).detach()

    def forward(self, x):
        return F.linear(x, self.ternarize(self.weight), self.bias)


@torch.no_grad()
def verify_ternary_constraint(model):
    bls = [m for m in model.modules() if isinstance(m, BitLinear)]
    for i, m in enumerate(bls):
        scale = m.weight.abs().mean().clamp(min=1e-5)
        w_tern = torch.round(m.weight / scale).clamp(-1, 1)
        uniq = set(w_tern.unique().tolist())
        assert uniq.issubset({-1.0, 0.0, 1.0}), f"BitLinear[{i}] levels={uniq}"
        assert torch.allclose(m.ternarize(m.weight), w_tern * scale, atol=1e-6)
    print(f"Ternary constraint OK: all {len(bls)} BitLinear layers in {{-1,0,1}}.")
    return True


# ---------------------------------------------------------------- Codec
class TDVQCodec(nn.Module):
    def __init__(self, d=8, bits=16, hidden=256, depth=2):
        super().__init__()
        self.d, self.bits = d, bits

        def stack(in_dim, out_dim):
            layers, dd = [], in_dim
            for _ in range(depth):
                layers += [BitLinear(dd, hidden), nn.GELU()]
                dd = hidden
            layers += [BitLinear(dd, out_dim)]
            return nn.Sequential(*layers)

        self.encoder = stack(d, bits)
        self.decoder = stack(bits, d)

    def encode_soft(self, v):
        return torch.sigmoid(self.encoder(v))

    def encode(self, v):
        """Hard B-bit codes (0/1)."""
        return (self.encode_soft(v) > 0.5).float()

    def decode(self, code):
        return self.decoder(code)

    def forward(self, v):
        p = self.encode_soft(v)
        hard = (p > 0.5).float()
        ste = p + (hard - p).detach()
        return self.decode(ste), ste


# ---------------------------------------------------------------- helpers
def vectorize(Wn, d):
    """[c, n] -> [n*c/d, d] along input dim (columns split into d-chunks)."""
    c, n = Wn.shape
    assert c % d == 0, f"in-dim {c} not divisible by d={d}"
    return Wn.t().reshape(n * (c // d), d)


def devectorize(V, c, n, d):
    return V.reshape(n, c).t().contiguous()


def col_normalize(W):
    s = W.abs().amax(dim=0).clamp(min=1e-8)  # [n]
    return W / s, s


def bits_per_weight(d, bits, c, scale_bits=16):
    return bits / d + scale_bits / c


# ---------------------------------------------------------------- training pool
@torch.no_grad()
def build_vector_pool(weight_dict, d, max_vectors=1_000_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    chunks = []
    for W in weight_dict.values():
        Wn, _ = col_normalize(W.float())
        chunks.append(vectorize(Wn, d))
    V = torch.cat(chunks, 0)
    if V.shape[0] > max_vectors:
        idx = torch.randperm(V.shape[0], generator=g)[:max_vectors]
        V = V[idx]
    return V


def train_codec(codec, V, epochs=30, batch=65536, lr=2e-3, device=DEV, log=True):
    codec.to(device)
    V = V.to(device)
    opt = torch.optim.Adam(codec.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    N = V.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        for i in range(0, N, batch):
            v = V[perm[i : i + batch]]
            vhat, _ = codec(v)
            loss = F.mse_loss(vhat, v)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * v.shape[0]
        sched.step()
        if log and (ep % 5 == 0 or ep == epochs - 1):
            with torch.no_grad():
                vhat, _ = codec(V)
                rel = ((vhat - V).pow(2).sum() / V.pow(2).sum()).sqrt().item()
            print(f"  codec ep{ep:3d} mse={tot/N:.6f} relRMSE={rel:.4f}")
    return codec


# ---------------------------------------------------------------- compress / decompress
@torch.no_grad()
def compress_matrix(codec, W, hdiag=None, refine_iters=0, device=DEV):
    """Return (codes[N,bits], scales[n], shape) and reconstruct W_hat."""
    codec.eval()
    c, n = W.shape
    d = codec.d
    Wn, s = col_normalize(W.float().to(device))
    V = vectorize(Wn, d)  # [N,d]
    codes = codec.encode(V)  # [N,bits]

    # activation-aware per-dim weights (diag Hessian), tiled to vectors
    if hdiag is not None:
        w = hdiag.to(device).clamp(min=1e-8)  # [c]
        wv = w.reshape(c // d, d).repeat(n, 1)  # [N,d] matches vectorize layout
    else:
        wv = None

    if refine_iters > 0:
        codes = _refine_codes(codec, V, codes, wv, refine_iters)

    Vhat = codec.decode(codes)
    Wn_hat = devectorize(Vhat, c, n, d)
    W_hat = Wn_hat * s
    return codes, s, (c, n), W_hat


@torch.no_grad()
def _refine_codes(codec, V, codes, wv, iters):
    """Greedy bit-flip refinement minimizing (weighted) reconstruction error."""
    B = codec.bits
    def werr(Vh):
        e = (Vh - V) ** 2
        if wv is not None:
            e = e * wv
        return e.sum(1)  # [N]
    cur = codes.clone()
    curerr = werr(codec.decode(cur))
    for _ in range(iters):
        improved = False
        for b in range(B):
            trial = cur.clone()
            trial[:, b] = 1 - trial[:, b]
            e = werr(codec.decode(trial))
            better = e < curerr
            if better.any():
                cur[better] = trial[better]
                curerr[better] = e[better]
                improved = True
        if not improved:
            break
    return cur


@torch.no_grad()
def decompress_matrix(codec, codes, s, shape):
    c, n = shape
    Vhat = codec.decode(codes.to(DEV))
    Wn_hat = devectorize(Vhat, c, n, codec.d)
    return Wn_hat * s.to(DEV)
