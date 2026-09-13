"""Premise check for D011: is D008's "33 pairs have a near-perfect query the
selection missed" a real finding or a winner's curse?

D008's oracle = max over all 259 queries of 2-way TEST accuracy, per pair. It was
selected on the evaluation split -- D008's own artifact says so ("an optimistic
bound") -- and with only 50 eval traces per pair, the max over 259 noisy
estimates is inflated by construction. Same failure mode D007/F2 corrected for,
and the same fix applies: SELECT on one half of the test configs, EVALUATE on the
other.
"""
import json
import numpy as np
import sys
sys.path[:0] = [".", "experiments"]
from d008_lib import build_dq, two_way_correct, load_tensor
import itertools

S, meta = load_tensor()
models = meta["models"]
pairs = list(itertools.combinations(models, 2))
fr = json.load(open("results/D008/identifiability_frontier.json"))
t2 = fr["T2"]["pairs"]
name2ix = {f"{a} | {b}": i for i, (a, b) in enumerate(pairs)}
t2ix = [name2ix[d["pair"]] for d in t2]
print(f"T2 pairs: {len(t2ix)}", flush=True)

Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")
rng = np.random.default_rng(20260913)
perm = rng.permutation(25)
hA, hB = set(perm[:12].tolist()), set(perm[12:24].tolist())
mA = np.isin(cfg_ev, list(hA)); mB = np.isin(cfg_ev, list(hB))

rows = []
for j, d in zip(t2ix, t2):
    a, b = pairs[j]
    ia, ib = models.index(a), models.index(b)
    accA, accB, acc_all = [], [], []
    for q in range(S.shape[0]):
        ok, em = two_way_correct(Dq[q], y_ev, y_rf, ia, ib)
        acc_all.append(ok.mean())
        accA.append(ok[mA[em]].mean()); accB.append(ok[mB[em]].mean())
    accA, accB, acc_all = map(np.array, (accA, accB, acc_all))
    qA = int(accA.argmax())                    # selected on half A
    rows.append(dict(pair=d["pair"], naive_oracle=float(acc_all.max()),
                     honest_oracle=float(accB[qA]),          # evaluated on half B
                     both_halves_max=float(np.minimum(accA, accB).max()),
                     median_q=float(np.median(acc_all)),
                     p95_q=float(np.percentile(acc_all, 95)),
                     build_qstar_acc=d["two_way_acc_best_query"],
                     reported_oracle=d["two_way_acc_oracle_over_pool"]))
    if len(rows) % 10 == 0:
        print(f"  {len(rows)}/{len(t2ix)}", flush=True)

n = np.array([r["naive_oracle"] for r in rows])
h = np.array([r["honest_oracle"] for r in rows])
bh = np.array([r["both_halves_max"] for r in rows])
med = np.array([r["median_q"] for r in rows])
print(f"\nnaive oracle (max over 259 on all 25 cfgs): mean {n.mean():.3f}, "
      f">=0.95 on {int((n>=0.95).sum())}/{len(rows)}")
print(f"HONEST oracle (pick on half A, score on half B): mean {h.mean():.3f}, "
      f">=0.95 on {int((h>=0.95).sum())}/{len(rows)}")
print(f"max over queries of min(halfA,halfB): mean {bh.mean():.3f}, "
      f">=0.95 on {int((bh>=0.95).sum())}/{len(rows)}")
print(f"median query accuracy: mean {med.mean():.3f}")
print(f"mean shrinkage naive->honest: {(n-h).mean():+.3f}")
json.dump(dict(n_pairs=len(rows),
               naive_oracle_ge95=int((n>=0.95).sum()),
               honest_oracle_ge95=int((h>=0.95).sum()),
               both_halves_ge95=int((bh>=0.95).sum()),
               naive_mean=float(n.mean()), honest_mean=float(h.mean()),
               both_halves_mean=float(bh.mean()),
               median_query_mean=float(med.mean()),
               shrinkage=float((n-h).mean()), split="12/12 of 25 test configs",
               pairs=rows), open("results/D011/premise_check.json","w"), indent=1)
print("written results/D011/premise_check.json")
