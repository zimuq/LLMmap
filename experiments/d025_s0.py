"""D025 / S0 — fix the standing D7 hard set H (DECISIONS.md D7, decided 2026-09-29).

Definition (D025 `## D`, verbatim):
  stratum N   the 65 structural near-relative pairs (d008_lib.near_relative_pairs)
  B0          paper8 (queries 0..7), k=8, D009 attention protocol, seeds 0-4,
              two-logit restricted accuracy per pair (logit_stats' per_hard_pair),
              evaluated on S_VAL, 5-seed mean
  H           {p in N : acc_B0(p) < 0.90}   (point estimate)
  bands       config bootstrap on S_val, boot_draws(25, 2000, seed=20260929),
              seeds averaged within a draw: stably hard if hi < 0.90, stably easy
              if lo > 0.90, else uncertain (descriptive; one sensitivity row)
  M_H         models in some pair of H
  F_H         M_H + every model joined to M_H by an N edge (one-step closure)
  groups      connected components of F_H under N edges

B0 checkpoints are reused (D009 r0, D016 r1-4). Before S_val is read, each is
reloaded and its S_test mean_top1 and all 65 per_hard_pair values asserted equal
to D009's stored runs. Descriptive only: paper8's S_test accuracy on the same
pairs, overlap with D019's consensus6, byte-identical greedy-reference answers per
H pair (D024 S0 arrays), and a cost projection for S2.
Stops: |H| = 0 -> stop and report; |H| > 12 -> stop and raise a Call.
Writes results/D025/hard_set.json.
Usage:  PYTHONPATH=.:experiments python experiments/d025_s0.py
"""
import os
import json
import time
import subprocess

import numpy as np

from d007_lib import load_corpus
from d008_lib import near_relative_pairs, boot_draws
from d009_lib import load_query_embeddings, build_traces, logit_stats
from d019_hard_model_recall import CONSENSUS6
from d023_train import reload_logits

OUT = "./results/D025"
TAU, K, N_SEEDS, NB, BOOT_SEED = 0.90, 8, 5, 2000, 20260929
CKPTS = ["./results/D009/models/paper8_k8_r0.ckpt"] + \
        [f"./results/D016/models/paper8_k8_r{r}.ckpt" for r in range(1, 5)]


def components(nodes, edges):
    nodes, adj = sorted(nodes), {n: set() for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        st, comp = [n], []
        seen.add(n)
        while st:
            x = st.pop(); comp.append(x)
            for y in adj[x] - seen:
                seen.add(y); st.append(y)
        comps.append(sorted(comp))
    return comps


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models = json.load(open("./results/D008/selection.json"))["models"]
    nm = len(models)
    hard = near_relative_pairs(models)
    keys = sorted(hard)
    assert len(keys) == 65
    names = [" | ".join(hard[i]["pair"]) for i in keys]

    d9 = [r for r in json.load(open("./results/D009/runs.json"))["runs"]
          if r["condition"] == "paper8" and r["k"] == K and not r["error"]]
    d9 = sorted(d9, key=lambda r: r["run"])
    assert len(d9) == N_SEEDS and all(r["queries"] == list(range(8)) for r in d9)

    qe = load_query_embeddings()
    q = list(range(8))
    tr_t, y_t, c_t, _ = build_traces(q, "test", qe, load_corpus(pool="test"))
    tr_v, y_v, c_v, _ = build_traces(q, "val", qe, load_corpus(pool="val"))

    val_cnt, val_pt, test_pp, repro = [], [], [], []
    for r, path in enumerate(CKPTS):
        assert os.path.exists(path), path
        lt = reload_logits(path, tr_t, nm, K)
        pt, _ = logit_stats(lt, y_t, c_t, hard, nm)
        ok_top1 = pt["mean_top1"] == d9[r]["mean_top1"]
        ok_pairs = pt["per_hard_pair"] == d9[r]["per_hard_pair"]
        repro.append(dict(seed=r, ckpt=path, mean_top1=pt["mean_top1"],
                          d009_mean_top1=d9[r]["mean_top1"], mean_top1_equal=ok_top1,
                          per_hard_pair_equal=ok_pairs))
        assert ok_top1 and ok_pairs, f"STOP: B0 seed {r} does not reproduce D009 ({path})"
        test_pp.append(pt["per_hard_pair"])
        lv = reload_logits(path, tr_v, nm, K)
        pv, sv = logit_stats(lv, y_v, c_v, hard, nm)
        val_pt.append(pv); val_cnt.append(sv["cnt_pair"])
    print(f"[S0] B0 reproduces D009 on S_test for all 5 seeds", flush=True)

    C = np.stack(val_cnt)                                     # (5, 65, 25)
    ncfg = C.shape[2]
    acc = C.sum(2).mean(0) / (2.0 * ncfg)                     # 5-seed mean, (65,)
    assert np.allclose(acc, np.mean([p["per_hard_pair"] for p in val_pt], 0), atol=6e-5)
    M = boot_draws(ncfg, NB, seed=BOOT_SEED)
    boot = (M @ C.mean(0).T) / (2.0 * M.sum(1)[:, None])      # (NB, 65)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    band = np.where(hi < TAU, "stably_hard", np.where(lo > TAU, "stably_easy", "uncertain"))
    test_acc = np.mean(test_pp, 0)

    H = [i for i, a in zip(keys, acc) if a < TAU]
    print(f"[S0] |H| = {len(H)}", flush=True)
    M_H = sorted({m for i in H for m in hard[i]["pair"]})
    n_edges = [hard[i]["pair"] for i in keys]
    F_H = set(M_H)
    for a, b in n_edges:
        if a in M_H or b in M_H:
            F_H.update((a, b))
    groups = components(F_H, n_edges)
    fam_pairs = [[f"{a} | {b}" for ai, a in enumerate(g) for b in g[ai + 1:]] for g in groups]
    group_h = [[names[keys.index(i)] for i in H if set(hard[i]["pair"]) <= set(g)] for g in groups]
    fam_is_spec = [len(g) == 2 and len(h) == 1 for g, h in zip(groups, group_h)]

    rows = []
    for j, i in enumerate(keys):
        rows.append(dict(pair=names[j], index_666=int(i), ab=list(hard[i]["ab"]),
                         same_base=hard[i]["same_base"], same_lineage=hard[i]["same_lineage"],
                         acc_val_B0=round(float(acc[j]), 4),
                         val_ci=[round(float(lo[j]), 4), round(float(hi[j]), 4)],
                         band=str(band[j]), in_H=bool(i in H),
                         acc_test_paper8_descriptive=round(float(test_acc[j]), 4)))

    # byte-identical greedy-reference answers (D024 S0 arrays), descriptive
    d24 = np.load("./results/D024/distances.npz")
    raw = d24["single_effective_raw"]
    med = np.median(raw, axis=1)
    identical = {names[keys.index(i)]: int((raw[:, i] <= 1e-6 * med).sum()) for i in H}

    # S2 cost projection: attention runs (GLOBAL k<=8 reused from D010)
    n_fam_new = sum(1 for f in fam_is_spec if not f)
    runs = N_SEEDS * (8 + 16 * len(H) + 16 * n_fam_new)
    proj_min = runs * 10.0 / 60.0          # ~7.4 s train (D024) + reload/extract, rounded up
    status = "OK" if 0 < len(H) <= 12 else ("STOP_EMPTY" if not H else "CALL_TOO_MANY")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    out = dict(
        schema="d025-hard-set-v1", status=status,
        definition=dict(
            source="DECISIONS.md D7 (human, 2026-09-29); D025 ## D",
            stratum_N="65 structural near-relative pairs, experiments/d008_lib.py:near_relative_pairs "
                      "(same base or same lineage, confs metadata; fixed since D001)",
            reference_protocol_B0="paper8 = queries [0..7], k=8, D009 attention protocol "
                                  "(confs/default.json inference_model, S_val early stopping), "
                                  "seeds 0-4, two-logit restricted accuracy per pair "
                                  "(d009_lib.logit_stats per_hard_pair), 5-seed mean",
            split="S_val (25 configs x 2 models = 50 traces per pair)", tau=TAU,
            membership="acc_B0(p) < tau (point estimate)",
            bands=f"config bootstrap on S_val, d008_lib.boot_draws(25, {NB}, seed={BOOT_SEED}), "
                  "seeds averaged within a draw; stably_hard hi<tau, stably_easy lo>tau, else "
                  "uncertain (descriptive; D025 secondary (f) only)",
            M_H="models in some pair of H",
            F_H="M_H plus every model joined to a member of M_H by an N edge (one-step closure)",
            family_groups="connected components of F_H under N edges",
            known_optimism="B0 checkpoints chosen by S_val loss: S_val accuracies slightly optimistic, "
                           "uniform across pairs"),
        provenance=dict(git_sha_at_compute=sha, b0_checkpoints=CKPTS,
                        b0_reproduces_d009_on_test=repro),
        H=[names[keys.index(i)] for i in H], H_index_666=[int(i) for i in H],
        H_stably_hard=[names[keys.index(i)] for i in H if band[keys.index(i)] == "stably_hard"],
        M_H=M_H, F_H=sorted(F_H), family_groups=groups, family_group_H_pairs=group_h,
        family_group_pairs=fam_pairs, family_group_equals_single_H_pair=fam_is_spec,
        counts=dict(H=len(H), M_H=len(M_H), F_H=len(F_H), groups=len(groups),
                    bands={b: int((band == b).sum()) for b in ("stably_hard", "uncertain", "stably_easy")},
                    H_by_band={b: int(sum(band[keys.index(i)] == b for i in H))
                               for b in ("stably_hard", "uncertain", "stably_easy")}),
        descriptive=dict(
            consensus6_overlap=sorted(set(M_H) & set(CONSENSUS6)),
            H_test_paper8={names[keys.index(i)]: round(float(test_acc[keys.index(i)]), 4) for i in H},
            test_paper8_below_tau=[names[j] for j in range(65) if test_acc[j] < TAU],
            byte_identical_greedy_answers_per_H_pair=identical),
        rows=rows,
        s2_cost_projection=dict(attention_runs=runs, est_gpu_min=round(proj_min, 1),
                                call_threshold_gpu_min=240,
                                note="GLOBAL k=9..16 + SPEC(p) k=1..16 per H pair + FAM(F) k=1..16 for "
                                     "groups that are not a single H pair; ~10 s/run incl. reload"),
        wall_s=round(time.time() - t0, 1))
    json.dump(out, open(f"{OUT}/hard_set.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: r["acc_val_B0"])[:15]:
        print(f"  {r['acc_val_B0']:.3f} {r['val_ci']} {r['band']:12s} test {r['acc_test_paper8_descriptive']:.3f} "
              f"{'H ' if r['in_H'] else '  '}{r['pair']}", flush=True)
    print(f"[S0] M_H {M_H}\n[S0] groups {groups}\n[S0] group H pairs {group_h}\n"
          f"[S0] counts {out['counts']}\n[S0] identical answers {identical}\n"
          f"[S0] cost {out['s2_cost_projection']}\n[S0] status {status}; wall {out['wall_s']} s",
          flush=True)
    if status != "OK":
        raise SystemExit(f"STOP: {status}")


if __name__ == "__main__":
    main()
