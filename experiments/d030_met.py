"""D030 — MET scorer (MET-Hamming; MET-e5 is added in P2).

`to_sequences` turns stored samples into MET's array layout with MET's own helpers
(`tokenize_unicode`, `pad_to_length`): column 0 = prompt id, then L = 1000 Unicode code
points padded with -1 (completions longer than L are cut to L, as MET's loader does).

`mmd_hamming_fast` is the block-diagonal form of the official `tests._mmd` with the Hamming
kernel: the kernel is zero across different prompts, every diagonal entry equals L (a
sequence matches itself at all L positions, padding included), so normalisation divides by
L, and the diagonal of K_XX / K_YY is excluded. S0.5(a) asserts it equals the official
`mmd_hamming(CompletionSample, CompletionSample)` to <= 1e-12.
"""
import numpy as np

L_CHARS = 1000


def to_sequences(samples, n_prompts, L=L_CHARS):
    from model_equality_testing.utils import tokenize_unicode, pad_to_length
    texts = [s["text"] for s in samples]
    arr = tokenize_unicode(texts)
    if arr.ndim == 1 or arr.shape[1] == 0:
        arr = np.full((len(texts), 1), -1)
    arr = arr[:, :L]
    arr = pad_to_length(arr, L=L)
    p = np.array([s["p"] for s in samples])[:, None]
    assert p.max() < n_prompts
    return np.concatenate([p, arr], axis=1)


def official(X, Y, n_prompts):
    import torch
    from model_equality_testing.distribution import CompletionSample
    from model_equality_testing.tests import mmd_hamming
    s1 = CompletionSample(prompts=torch.tensor(X[:, 0]), completions=torch.tensor(X[:, 1:]), m=n_prompts)
    s2 = CompletionSample(prompts=torch.tensor(Y[:, 0]), completions=torch.tensor(Y[:, 1:]), m=n_prompts)
    return float(mmd_hamming(s1, s2))


def mmd_hamming_fast(X, Y):
    L = X.shape[1] - 1
    sxx = sxy = syy = 0.0
    nxx = nxy = nyy = 0
    for p in np.union1d(np.unique(X[:, 0]), np.unique(Y[:, 0])):
        a, b = X[X[:, 0] == p, 1:], Y[Y[:, 0] == p, 1:]
        kxx = (a[:, None, :] == a[None, :, :]).sum(-1).astype(float) / L
        kyy = (b[:, None, :] == b[None, :, :]).sum(-1).astype(float) / L
        kxy = (a[:, None, :] == b[None, :, :]).sum(-1).astype(float) / L
        sxx += kxx.sum() - np.trace(kxx); nxx += len(a) * (len(a) - 1)
        syy += kyy.sum() - np.trace(kyy); nyy += len(b) * (len(b) - 1)
        sxy += kxy.sum(); nxy += len(a) * len(b)
    return sxx / nxx - 2 * sxy / nxy + syy / nyy
