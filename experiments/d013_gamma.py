"""D013 / S1-S3 — verify D008's frozen GreedyCover chains, and check the
structural prediction at the tensor level before any training.

S2 asks to "run GreedyCover selection at gamma in {0.5, 0.25, 0.05}". Those
chains ALREADY EXIST, frozen, in results/D008/gamma_sweep.json. Under I6 the
right move is to read them -- but "already recorded" is not "verified", so this
recomputes all five from the frozen tensor and asserts they match. GreedyCover
is cheap enough (D008 measured 0.02 s per k=8 chain) that verifying all five
costs less than reading the file.
"""
import os, json, time
import numpy as np
import sys
sys.path[:0] = [".", "experiments"]
from LLMmap.greedy_cover import greedy_cover, cvar
from d008_lib import load_tensor

OUT = "./results/D013"; os.makedirs(OUT, exist_ok=True)
S, meta = load_tensor()
g8 = json.load(open("results/D008/gamma_sweep.json"))
d8 = json.load(open("results/D008/selection.json"))
GAMMAS = ["1.0", "0.5", "0.25", "0.1", "0.05"]

chains, traces, wall = {}, {}, {}
for g in GAMMAS:
    t = time.time()
    sel, tr = greedy_cover(S, 8, gamma=float(g), agg="max", return_trace=True)
    wall[g] = time.time() - t
    chains[g], traces[g] = sel, tr
    frozen = g8["chains"][g]
    ok = sel == frozen
    print(f"[S2] gamma={g:5s} {sel}  matches D008 frozen: {ok}  ({wall[g]:.3f}s)",
          flush=True)
    assert ok, f"gamma={g} chain diverged from D008's frozen record: {sel} vs {frozen}"

assert chains["1.0"] == d8["conditions"]["mean_greedy_max"]["queries"][:8]
assert chains["0.1"] == d8["conditions"]["cvar_max"]["queries"][:8]
print("[S2] gamma=1.0/0.1 chains == the ones D009 actually trained: True", flush=True)

print("\n[S3] STRUCTURAL PREDICTION: is GreedyCover's objective monotone in k?")
mono = {}
for g in GAMMAS:
    o = [t["objective"] for t in traces[g]]
    mono[g] = all(o[i] <= o[i+1] + 1e-12 for i in range(len(o)-1))
    print(f"  gamma={g:5s} {[round(x,4) for x in o]}  monotone: {mono[g]}")

d12 = json.load(open("results/D012/objective_traces.json"))
jg1 = [t["objective_cvar"] for t in d12["objective_traces"]["1.0"]]
print(f"\n  JointGreedy gamma=1.0 (D012, for contrast): {[round(x,4) for x in jg1]}")
print(f"  GreedyCover gamma=1.0:                      "
      f"{[round(t['objective'],4) for t in traces['1.0']]}")
print(f"  -> both start at k=1 identical ({jg1[0]:.4f}); GreedyCover RISES to "
      f"{traces['1.0'][-1]['objective']:.4f}, JointGreedy FALLS to {jg1[-1]:.4f}")

json.dump(dict(chains=chains,
               traces={g: [{k: v for k, v in t.items()} for t in traces[g]]
                       for g in GAMMAS},
               monotone=mono, wall_s={g: round(w, 4) for g, w in wall.items()},
               verified_against="results/D008/gamma_sweep.json (all five match)",
               d009_trained_points=["1.0 (mean_greedy_max)", "0.1 (cvar_max)"],
               jointgreedy_gamma1_trace=jg1),
          open(f"{OUT}/gamma_selection.json", "w"), indent=1)
print(f"\nwritten {OUT}/gamma_selection.json")
