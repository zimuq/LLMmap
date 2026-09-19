"""D014 / S1 — tensor-level chain analysis at a fine gamma grid, both algorithms.

Free, and explicitly to be reported BEFORE any training. The question it can
settle on its own: do the gammas between 0.25 and 1.0 produce chains that are
gamma=1-like, gamma=0.25-like, or genuinely their own? If selection "flips"
between two chain families at some gamma, the step is visible here and the
training only has to confirm it.
"""
import os, json, time, itertools
import numpy as np
import sys
sys.path[:0] = [".", "experiments"]
from LLMmap.greedy_cover import greedy_cover
from LLMmap.joint_greedy import joint_greedy, peak_k
from d008_lib import load_tensor, near_relative_pairs
from d007_lib import load_corpus
from d010_select import precompute

OUT = "./results/D014"; os.makedirs(OUT, exist_ok=True)
GRID = [1.0, 0.75, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.1, 0.05]
K = 8

S, meta = load_tensor()
models = meta["models"]
npairs = S.shape[1]
print(f"tensor {S.shape}; CVaR count = floor(gamma*{npairs}):", flush=True)
print("  " + "  ".join(f"g={g}:{max(1,int(np.floor(g*npairs)))}" for g in GRID),
      flush=True)

# ---- GreedyCover
gc, gc_tr = {}, {}
t = time.time()
for g in GRID:
    sel, tr = greedy_cover(S, K, gamma=g, agg="max", return_trace=True)
    gc[str(g)], gc_tr[str(g)] = sel, tr
print(f"\n[GreedyCover] {time.time()-t:.1f}s")
for g in GRID:
    print(f"  g={g:<5} {gc[str(g)]}")

# ---- JointGreedy
pairs = list(itertools.combinations(models, 2))
ix = {m: i for i, m in enumerate(models)}
ia = np.array([ix[a] for a, _ in pairs]); ib = np.array([ix[b] for _, b in pairs])
_, X = load_corpus(pool="build")
t = time.time(); XX2, XY2 = precompute(X, models, pairs)
print(f"\n[JointGreedy] precompute {(time.time()-t)/60:.2f} min", flush=True)
jg, jg_tr = {}, {}
t = time.time()
for g in GRID:
    sel, tr = joint_greedy(XY2, XX2, ia, ib, K, g, "energy", verbose=False)
    jg[str(g)], jg_tr[str(g)] = sel, tr
    print(f"  g={g:<5} {sel}  peak k={peak_k(tr)[0]}", flush=True)
print(f"[JointGreedy] {(time.time()-t)/60:.2f} min")

def report(name, chains, traces):
    print(f"\n===== {name}: chain identity and overlap =====")
    # identity groups
    groups = {}
    for g in GRID:
        key = tuple(chains[str(g)])
        groups.setdefault(key, []).append(g)
    print(f"  distinct k=8 CHAINS (order-sensitive): {len(groups)}")
    sets = {}
    for g in GRID:
        key = tuple(sorted(chains[str(g)]))
        sets.setdefault(key, []).append(g)
    print(f"  distinct k=8 SETS (order-insensitive): {len(sets)}")
    for key, gg in sets.items():
        if len(gg) > 1:
            print(f"    identical set for gamma in {gg}")
    print(f"  {'gamma':>6s} {'ov vs g=1.0':>12s} {'ov vs g=0.25':>13s} "
          f"{'k=4 vs 1.0':>11s} {'k=4 vs 0.25':>12s} {'first q':>8s}")
    for g in GRID:
        c = chains[str(g)]
        o1 = len(set(c) & set(chains["1.0"])); o2 = len(set(c) & set(chains["0.25"]))
        f1 = len(set(c[:4]) & set(chains["1.0"][:4]))
        f2 = len(set(c[:4]) & set(chains["0.25"][:4]))
        print(f"  {g:>6} {o1:>12d} {o2:>13d} {f1:>11d} {f2:>12d} {c[0]:>8d}")
    return {"distinct_chains": len(groups), "distinct_sets": len(sets),
            "set_groups": {str(gg): list(k) for k, gg in
                           [(k, v) for k, v in sets.items()]}}

r_gc = report("GreedyCover", gc, gc_tr)
r_jg = report("JointGreedy", jg, jg_tr)

json.dump(dict(grid=GRID, n_pairs=npairs,
               cvar_count={str(g): max(1, int(np.floor(g*npairs))) for g in GRID},
               greedycover=dict(chains=gc, traces=gc_tr, summary=r_gc,
                                monotone={g: all(t[i]["objective"] <= t[i+1]["objective"]+1e-12
                                                 for i in range(len(t)-1))
                                          for g, t in gc_tr.items()}),
               jointgreedy=dict(chains=jg, traces=jg_tr, summary=r_jg,
                                peak_k={g: peak_k(t)[0] for g, t in jg_tr.items()})),
          open(f"{OUT}/chain_analysis.json", "w"), indent=1)
print(f"\nwritten {OUT}/chain_analysis.json")
