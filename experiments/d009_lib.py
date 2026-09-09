"""D009 — trace assembly and evaluation for LLmap's own trained stage-2 network.

A trace, per `METHOD.md §5.5` and `LLMmap/inference.py:105-113`, is

    trace = [ E(query) ; E(response) ]          2048-d, per query slot
            stacked into a length-k sequence    (k, 2048)

D004-D008 stored response embeddings only (1024-d). That was correct there --
at fixed `q`, `E(query)` is the same constant for every model compared, so it
cancels in any distance statistic. It does NOT cancel here: the self-attention
network needs it to know *which* pool query produced each slot.

WHY THIS BYPASSES `LLMmap/dataset.py`'s `load_datasets` (disclosed in P1).
The shipped loader takes raw text and re-embeds it through `EmbeddingCache`.
We already hold every response embedding from D006/S6 and every pool-query
embedding from the same run, computed by the same frozen I5 model with the same
pooling and the same 512-token window. Re-embedding would (a) cost a GPU pass
per condition x k, (b) risk silently producing embeddings that differ from the
ones D006/D007/D008's numbers were computed on, which is exactly what I6
exists to prevent. So the assembly here reads the frozen arrays and produces
the identical tensor the shipped loader would have produced -- and
`d009_pilot.py` checks that claim numerically against the shipped path rather
than asserting it.

The TRAINING LOOP is not bypassed: `LLMmap/trainer.py` and
`LLMmap/inference_model_archs.py` are used unmodified.
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from d007_lib import load_corpus
from d008_lib import near_relative_pairs, OUT as D008_OUT

CORPUS = os.environ.get("D006_CORPUS", "./data/corpus_v1")
EMB = os.path.join(CORPUS, "embeddings")
QEMB = os.path.join(EMB, "_queries.npy")
POOL = "./confs/queries/pool_v1.json"
OUT = "./results/D009"


def load_query_embeddings():
    """The 259 pool-query embeddings, produced by D006/S6 under the same I5
    model as every response embedding. S1 is therefore already satisfied by a
    frozen artifact; this verifies it rather than recomputing it."""
    qe = np.load(QEMB)
    idx = json.load(open(os.path.join(EMB, "_queries.index.json")))
    pool = json.load(open(POOL))
    assert qe.shape == (259, 1024), qe.shape
    assert idx["n"] == 259 and idx["dim"] == 1024
    assert idx["embedding_model"] == "intfloat/multilingual-e5-large-instruct"
    assert idx["q0_sha256"] == pool["sha256"], (
        "query embeddings were computed against a different Q_0 than the pool "
        "on disk -- do not proceed, the slot->text mapping would be wrong")
    return qe.astype(np.float32)


def build_traces(queries, pool_name, qe=None, cubes=None):
    """(N, k, 2048) float32 traces, labels, and config ids for one split.

    N = 37 models x n_configs(pool). Slot order follows `queries` exactly, so
    two strategies selecting the same queries in a different order produce
    different (equally valid) sequences -- the network is permutation-sensitive
    only through its positional content, which here is E(query) itself.
    """
    qe = load_query_embeddings() if qe is None else qe
    models, X = load_corpus(pool=pool_name) if cubes is None else cubes
    nq, nc, dim = X[models[0]].shape
    k = len(queries)

    tr = np.empty((len(models) * nc, k, 2 * dim), np.float32)
    y = np.repeat(np.arange(len(models)), nc)
    cfg = np.tile(np.arange(nc), len(models))
    for mi, m in enumerate(models):
        cube = X[m]
        for si, q in enumerate(queries):
            r = mi * nc
            tr[r:r + nc, si, :dim] = qe[q]                    # E(query)
            tr[r:r + nc, si, dim:] = cube[q].astype(np.float32)  # E(response)
    return tr, y, cfg, models


def make_loader(tr, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(tr), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def hparams_from_shipped(k, n_models, path="./confs/default.json"):
    """The shipped defaults, verbatim. D009/S3 requires the source be stated:
    it is `confs/default.json`'s `inference_model` block, which is the released
    repo's own configuration. Only the three fields the loader would have filled
    in from the data (`num_classes`, `num_queries`, `emb_size`) are set here."""
    conf = json.load(open(path))
    hp = dict(conf["inference_model"])
    hp["num_classes"] = n_models
    hp["num_queries"] = k
    hp["emb_size"] = 1024
    return hp, conf


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def logits_for(model, tr, device, batch=256):
    model.eval().to(device)
    out = []
    for i in range(0, len(tr), batch):
        x = torch.from_numpy(tr[i:i + batch]).to(device)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def logit_stats(logits, y, cfg, hard_pairs, n_models):
    """The four METHOD.md §8 metrics from closed-set logits, plus the
    per-config count structures D008's bootstrap consumes unchanged.

    hard-subset is D008/Call-2's definition carried over: 2-way restricted
    accuracy on the 65 structural near-relative pairs. For a 37-way classifier
    the natural restriction is argmax over just those two classes' logits --
    the same question ("which of these two is it?") the D008 proxy asked of a
    2-class reference set.
    """
    ncfg = int(cfg.max()) + 1
    pred = logits.argmax(1)
    correct = pred == y

    cnt_total = np.bincount(cfg[correct], minlength=ncfg).astype(np.float64)
    cnt_model = np.zeros((n_models, ncfg), np.float64)
    np.add.at(cnt_model, (y[correct], cfg[correct]), 1.0)

    keys = sorted(hard_pairs)
    cnt_pair = np.zeros((len(keys), ncfg), np.float64)
    hv = np.empty(len(keys))
    for r, i in enumerate(keys):
        a, b = hard_pairs[i]["ab"]
        m = (y == a) | (y == b)
        two = logits[m][:, [a, b]]
        p2 = np.where(two.argmax(1) == 0, a, b)
        ok = p2 == y[m]
        cnt_pair[r] = np.bincount(cfg[m][ok], minlength=ncfg)
        hv[r] = ok.mean()

    per_model = cnt_model.sum(1) / (correct.size / n_models)
    point = dict(
        mean_top1=float(correct.mean()),
        worst_class=float(per_model.min()),
        worst3_class=float(np.sort(per_model)[:3].mean()),
        hard_subset=float(hv.mean()),
        hard_subset_min=float(hv.min()),
        per_model=per_model.round(4).tolist(),
        per_hard_pair=hv.round(4).tolist())
    return point, dict(cnt_total=cnt_total, cnt_model=cnt_model,
                       cnt_pair=cnt_pair, n_models=n_models, ncfg=ncfg)


def d008_chains():
    """D008's selected chains, verbatim (I6 applies to the selection outputs)."""
    sel = json.load(open(f"{D008_OUT}/selection.json"))
    draws = json.load(open(f"{D008_OUT}/random_draws.json"))
    return dict(
        cvar_max=sel["conditions"]["cvar_max"]["queries"],
        mean_greedy_max=sel["conditions"]["mean_greedy_max"]["queries"],
        paper8=list(range(8)),
    ), draws, sel


# ------------------------------------------------------- open-set (S8) pairs
class TracePairDataset(torch.utils.data.Dataset):
    """Siamese pair sampler over assembled traces.

    Mirrors `LLMmap/dataset.py`'s `DatasetFactorySiamese` semantics exactly --
    per-index reseeding, a 50/50 positive/negative draw, and **label 1 = same
    model**. That labelling looks inverted against `ContrastiveLoss`'s docstring
    ("0 = same"), but the docstring describes the classic distance-based form
    while the network here emits a SIMILARITY (`sigmoid(fc(bn(dist)))`). Under
    that output: y=1 gives `(margin - y_pred)^2`, pushing similarity up for same
    -model pairs, and y=0 gives `y_pred^2`, pushing it down for different ones.
    The shipped code is self-consistent; only its docstring is misleading.
    """
    def __init__(self, tr, y, n_pairs, n_models):
        self.tr, self.y, self.n_pairs, self.n_models = tr, y, n_pairs, n_models
        self.by_model = [np.where(y == m)[0] for m in range(n_models)]

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        import random as _r
        info = torch.utils.data.get_worker_info()
        _r.seed(idx + (0 if info is None else info.id))
        a = _r.randrange(0, self.n_models)
        ia = _r.choice(list(self.by_model[a]))
        if _r.choice([True, False]):
            pool = [i for i in self.by_model[a] if i != ia]
            ib, label = _r.choice(pool), 1
        else:
            b = _r.choice([m for m in range(self.n_models) if m != a])
            ib, label = _r.choice(list(self.by_model[b])), 0
        pair = np.stack([self.tr[ia], self.tr[ib]])
        return torch.from_numpy(pair), float(label)
