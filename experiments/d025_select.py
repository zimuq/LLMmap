"""D025 / S1 — chains, frozen before any training (D025 Review 2026-09-29, APPROVED
WITH AMENDMENTS).

Code-path check: select(all 666, 8) == D010's chain (stop condition).
Arms: GLOBAL to k=16 (prefix 1..8 == D010, asserted); SPEC(p) per H pair; FAM(F)
per family group; a FAM whose group is exactly one H pair must equal that
SPEC chain (asserted; the FAM arm is then not trained separately).
Also recorded, target-free: objective traces + peak k, overlaps with GLOBAL per k,
identical-input k vs GLOBAL, D022 union flags, and the Review's pre-training
CEILING / attribution tables (from GLOBAL's existing k=8 numbers: D017 linear
concat per-pair, D010 attention per_hard_pair 5-seed mean).
Writes results/D025/selection.json with frozen_before_training = true.
Usage:  PYTHONPATH=.:experiments python experiments/d025_select.py
"""
import json
import time
import hashlib

import numpy as np

from LLMmap.joint_greedy import peak_k
from d007_lib import load_corpus
from d010_select import precompute
from d025_lib import OUT, K_MAX, D010_CHAIN, arm_definitions, select

M_MARGIN = 0.05


def main():
    t0 = time.time()
    models, X = load_corpus(pool="build")
    assert models == json.load(open("./results/D008/selection.json"))["models"]
    arms, pairs, hs = arm_definitions(models)
    ix = {m: i for i, m in enumerate(models)}
    ia = np.array([ix[a] for a, _ in pairs])
    ib = np.array([ix[b] for _, b in pairs])
    XX2, XY2 = precompute(X, models, pairs)
    del X

    # code-path check (stop condition)
    chk, _ = select(XY2, XX2, ia, ib, list(range(len(pairs))), k=8)
    assert chk == D010_CHAIN, f"STOP: code-path check {chk} != {D010_CHAIN}"
    print(f"[S1] code-path check OK: {chk}", flush=True)

    chains, traces = {}, {}
    for name, a in arms.items():
        sel, tr = select(XY2, XX2, ia, ib, a["P"], k=K_MAX)
        chains[name], traces[name] = sel, tr
        print(f"[S1] {name} (|P|={len(a['P'])}): {sel}  peak k={peak_k(tr)[0]}", flush=True)
    assert chains["GLOBAL"][:8] == D010_CHAIN

    fam_equals_spec, trained = {}, []
    for name, a in arms.items():
        if a["kind"] == "FAM" and len(a["P"]) == 1:
            spec = next(s for s, b in arms.items() if b["kind"] == "SPEC" and b["P"] == a["P"])
            assert chains[name] == chains[spec], (name, spec)
            fam_equals_spec[name] = spec
        else:
            trained.append(name)

    g = chains["GLOBAL"]
    ident = {n: [k for k in range(1, K_MAX + 1) if chains[n][:k] == g[:k]] for n in trained if n != "GLOBAL"}
    overlap = {n: [len(set(chains[n][:k]) & set(g[:k])) for k in range(1, K_MAX + 1)] for n in chains}
    flags = {r["id"]: r for r in json.load(open("./results/D022/query_flags.json"))["rows"]}
    flag_ct = {n: {k: int(sum(bool(flags[q]["union_flag"]) for q in c[:k])) for k in (8, 16)}
               for n, c in chains.items()}

    # pre-training CEILING / attribution (Review Call 1), from existing GLOBAL k=8 numbers
    rows = {r["pair"]: r for r in hs["rows"]}
    keys = [r["pair"] for r in hs["rows"]]
    lin = json.load(open("./results/D017/linear_per_hard_pair.json"))["concat"]["joint_energy"]["8"]
    d10 = [r for r in json.load(open("./results/D010/metrics_by_k.json"))["runs"]
           if r["condition"] == "joint_energy" and r["k"] == 8 and not r["error"]]
    assert len(d10) == 5
    att = np.mean([r["per_hard_pair"] for r in d10], 0)
    ceiling = {}
    for p in hs["H"]:
        lv, av = float(lin[p]), float(att[keys.index(p)])
        cl, ca = (1 - lv) < M_MARGIN, (1 - av) < M_MARGIN
        attrib = ("NOT HARD ON S_test" if cl and ca else
                  "READOUT-LIMITED" if cl and not ca else "tested")
        ceiling[p] = dict(linear_global_k8=round(lv, 4), attention_global_k8=round(av, 4),
                          ceiling_linear=bool(cl), ceiling_attention=bool(ca), attribution=attrib)
    print(f"[S1] ceiling/attribution: {ceiling}", flush=True)

    out = dict(
        schema="d025-selection-v1", frozen_before_training=True,
        review="D025 Review 2026-09-29, APPROVED WITH AMENDMENTS (Call 1 CEILING + attribution; "
               "linear primary; A1, A2; Calls 2-4)",
        hard_set_sha256=hashlib.sha256(open(f"{OUT}/hard_set.json", "rb").read()).hexdigest(),
        code_path_check=dict(chain=chk, d010=D010_CHAIN, identical=True),
        arms={n: dict(kind=a["kind"], n_pairs=len(a["P"]), target=a["target"],
                      models=a.get("models"), P=a["P"] if len(a["P"]) < 666 else "all 666")
              for n, a in arms.items()},
        chains=chains, traces=traces,
        peak_k={n: peak_k(t)[0] for n, t in traces.items()},
        fam_equals_spec=fam_equals_spec, trained_arms=trained,
        train_order="each chain's own order (D convention); k-prefix of the chain",
        identical_input_k_vs_GLOBAL=ident, overlap_with_GLOBAL_by_k=overlap,
        d022_union_flags=flag_ct,
        ceiling_and_attribution=dict(
            rule="CEILING_r(p): 1 - acc_r,GLOBAL(p,8) < 0.05, fixed from existing GLOBAL k=8 point "
                 "estimates before any SPEC/FAM training (Review Call 1)",
            sources=dict(linear="results/D017/linear_per_hard_pair.json concat joint_energy k=8",
                         attention="results/D010/metrics_by_k.json joint_energy k=8 per_hard_pair, "
                                   "5-seed mean"),
            pairs=ceiling,
            primary_linear_population=[p for p, v in ceiling.items() if not v["ceiling_linear"]],
            secondary_attention_population=[p for p, v in ceiling.items() if not v["ceiling_attention"]]),
        wall_s=round(time.time() - t0, 1))
    json.dump(out, open(f"{OUT}/selection.json", "w"), indent=1)
    print(f"[S1] identical-input k vs GLOBAL: {ident}\n[S1] flags {flag_ct}\n"
          f"[S1] primary linear population {out['ceiling_and_attribution']['primary_linear_population']}; "
          f"attention {out['ceiling_and_attribution']['secondary_attention_population']}\n"
          f"[S1] wrote {OUT}/selection.json; wall {out['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
