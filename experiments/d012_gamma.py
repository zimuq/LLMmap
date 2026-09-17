"""D012 / S2 (authorized cost+sanity check) + a direct test of the pre-registered
mechanism, at the tensor level, before any training.

D012's hypothesis: the joint statistic under MEAN aggregation may already
implicitly favor hard pairs, making gamma less necessary than it was for
GreedyCover. S4 infers this indirectly from objective traces. It can be measured
directly and for free: under each gamma, look at where the selected chain's
per-pair joint statistic sits for HARD (near-relative) pairs vs the rest. If
mean-aggregation already lifts hard pairs, the gamma=1 chain's hard-pair
percentile should already be high and gamma should move it little.
"""
import os, json, time, itertools
import numpy as np
import sys
sys.path[:0] = [".", "experiments"]
from LLMmap.joint_greedy import joint_greedy, peak_k
from LLMmap.joint_statistic import cvar
from d008_lib import near_relative_pairs
from d007_lib import load_corpus
from d010_select import precompute

OUT = "./results/D012"
os.makedirs(OUT, exist_ok=True)
models, X = load_corpus(pool="build")
pairs = list(itertools.combinations(models, 2))
ix = {m: i for i, m in enumerate(models)}
ia = np.array([ix[a] for a, _ in pairs]); ib = np.array([ix[b] for _, b in pairs])
hard = near_relative_pairs(models)
hmask = np.zeros(len(pairs), bool); hmask[list(hard)] = True
print(f"{len(pairs)} pairs, {hmask.sum()} near-relative", flush=True)

t = time.time(); XX2, XY2 = precompute(X, models, pairs); t_pre = time.time()-t
print(f"[S2] precompute {t_pre/60:.2f} min (D010 measured 0.2)", flush=True)

def per_pair_stat(qs):
    xy = XY2[qs].sum(0); xxm = XX2[qs].sum(0)
    xy_ = np.sqrt(xy).mean(axis=(1,2)); xm = np.sqrt(xxm).mean(axis=(1,2))
    xx, yy = xm[ia], xm[ib]
    within = (xx+yy)/2.0
    return (2*xy_-xx-yy)/np.maximum(within,1e-12)

res = {}
d010 = json.load(open("results/D010/selection.json"))
chains = {"0.1": d010["joint_energy"]["queries"][:8]}
for g in (1.0, 0.25, 0.05):
    t = time.time()
    sel, tr = joint_greedy(XY2, XX2, ia, ib, 8, g, "energy", verbose=False)
    chains[str(g)] = sel
    pk, pv = peak_k(tr)
    res[str(g)] = dict(queries=sel, trace=tr, peak_k=pk, wall_min=round((time.time()-t)/60,2))
    print(f"[S2/S3] gamma={g}: {sel}  peak k={pk}  {(time.time()-t)/60:.2f} min", flush=True)

print("\n[mechanism] where do HARD pairs sit under each chain?")
mech = {}
for g, qs in sorted(chains.items(), key=lambda kv: -float(kv[0])):
    s = per_pair_stat(qs)
    hp = s[hmask]; ep = s[~hmask]
    pct = float(np.mean([(s < v).mean() for v in hp])*100)
    mech[g] = dict(hard_mean=float(hp.mean()), easy_mean=float(ep.mean()),
                   hard_mean_percentile=pct, cvar10=cvar(s,0.1), mean=float(s.mean()),
                   frac_hard_in_worst_decile=float(hmask[np.argsort(s)[:67]].mean()))
    print(f"  gamma={g:5s} hard mean {hp.mean():.4f} vs easy {ep.mean():.4f}  "
          f"hard@pct {pct:5.1f}  CVaR10 {cvar(s,0.1):.4f}  mean {s.mean():.4f}  "
          f"hard-share of worst decile {mech[g]['frac_hard_in_worst_decile']:.2f}")
ov = {}
gs = sorted(chains, key=lambda x: -float(x))
for i in range(len(gs)):
    for j in range(i+1, len(gs)):
        ov[f"{gs[i]}_vs_{gs[j]}"] = len(set(chains[gs[i]]) & set(chains[gs[j]]))
print(f"\n[S4] chain overlap at k=8: {ov}")
json.dump(dict(precompute_min=round(t_pre/60,2), chains=chains, selections=res,
               mechanism=mech, chain_overlap_k8=ov,
               n_hard=int(hmask.sum()), n_pairs=len(pairs),
               note="base rate of near-relative pairs is 65/666 = 9.8%"),
          open(f"{OUT}/s2_check.json","w"), indent=1)
print(f"written {OUT}/s2_check.json")
