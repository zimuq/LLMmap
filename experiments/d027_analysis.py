"""D027 / S3 — instrument control, H' (D7 rule on the v2 universe), label,
secondaries (a)-(f), and the frozen v2 universe.

All numbers are S_val (no S_test exists). Two-logit restricted accuracy per
pair uses logit_stats' rule (argmax over [a, b]; a tie goes to a). Bands:
config bootstrap on S_val, d008_lib.boot_draws(25, 2000, seed=20261002), seeds
averaged within a draw (D025 S0's method). Flags per N' pair: mixed_env
(Review A1), batch1 (second Review), and the members' (f1)/tokparity status.

Usage:  PYTHONPATH=.:experiments python experiments/d027_analysis.py
"""
import csv
import json
import time
import itertools

import numpy as np
from scipy.stats import spearmanr

from d008_lib import boot_draws, summed, two_way_correct
from d025_s0 import components

RES = "./results/D027"
SCR = "./data/corpus_v2_screen"
TAU, NB, BOOT_SEED = 0.90, 2000, 20261002
CONTROL = ["tiiuae/Falcon3-10B-Instruct | tiiuae/Falcon3-7B-Instruct",
           "microsoft/Phi-3-medium-128k-instruct | microsoft/Phi-3-medium-4k-instruct"]


def two_logit_counts(lg, y, cfg, a, b, ncfg=25):
    out = np.zeros(ncfg)
    for d, m in enumerate((a, b)):
        sel = y == m
        ok = lg[sel, a] >= lg[sel, b] if d == 0 else lg[sel, b] > lg[sel, a]
        out += np.bincount(cfg[sel][ok], minlength=ncfg)
    return out


def main():
    t0 = time.time()
    runs = json.load(open(f"{RES}/b0_runs.json"))
    models = runs["models"]
    ix = {m: i for i, m in enumerate(models)}
    nm = len(models)
    L = np.load(f"{RES}/logits_b0.npz")
    y, cfg = L["y"], L["cfg"]
    seeds = [L[f"b0_r{r}"] for r in range(5)]
    lin = np.load(f"{RES}/logits_linear.npz")["linear"]

    v1 = sorted(s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"])
    meta = {r["model"]: r for r in csv.DictReader(open("./results/D001/model_metadata.csv"))}
    meta.update({r["model"]: r for r in csv.DictReader(open(f"{RES}/candidates_metadata.csv"))})
    npr = json.load(open(f"{RES}/n_prime.json"))
    assert npr["n_universe"] == nm
    N = [p["pair"] for p in npr["pairs"]]
    man = json.load(open(f"{RES}/manifest.json"))["models"]
    s0 = json.load(open(f"{RES}/s0.json"))
    tokp = json.load(open(f"{RES}/tokparity.json"))
    env = {m: man[m]["gen_env"] for m in models}
    b1 = {m: bool(man[m]["batch1"]) for m in models}

    M = boot_draws(25, NB, seed=BOOT_SEED)

    def pair_stats(pair):
        a, b = (ix[m] for m in pair.split(" | "))
        C = np.stack([two_logit_counts(lg, y, cfg, a, b) for lg in seeds])       # (5, 25)
        acc = float(C.sum(1).mean() / 50.0)
        bt = (M @ C.mean(0)) / (2.0 * M.sum(1))
        lo, hi = np.percentile(bt, [2.5, 97.5])
        band = "stably_hard" if hi < TAU else ("stably_easy" if lo > TAU else "uncertain")
        lin_acc = float(two_logit_counts(lin, y, cfg, a, b).sum() / 50.0)
        return dict(acc=acc, ci=[float(lo), float(hi)], band=band, linear=lin_acc,
                    per_seed=(C.sum(1) / 50.0).round(4).tolist())

    rows = {}
    for p in N:
        a, b = p.split(" | ")
        st = pair_stats(p)
        st.update(
            pair=p, candidate_members=[m for m in (a, b) if m not in v1],
            lineages=sorted({meta[a]["lineage"], meta[b]["lineage"]}),
            mixed_env=env[a] != env[b], gen_env=[env[a], env[b]],
            batch1=b1[a] != b1[b], batch1_members=[m for m in (a, b) if b1[m]],
            f1_identical_rate={m: man[m]["f1"]["identical_rate"] for m in (a, b) if man[m].get("f1")},
            tokparity={m: ("identical" if not (tokp.get(m.replace("/", "__"), {}).get("ids_differ")
                                               or tokp.get(m.replace("/", "__"), {}).get("text_differs"))
                           else "differs") for m in (a, b) if m in v1})
        rows[p] = st

    # ---- instrument control (before the label)
    d25 = json.load(open("./results/D025/hard_set.json"))
    d25acc = {r["pair"]: r["acc_val_B0"] for r in d25["rows"]}
    common = [p for p in d25acc if p in rows]
    assert len(common) == 65
    rho = spearmanr([d25acc[p] for p in common], [rows[p]["acc"] for p in common]).correlation
    control = dict(
        pairs={p: dict(acc_v2=rows[p]["acc"], ci=rows[p]["ci"], acc_d025=d25acc[p], below_tau=rows[p]["acc"] < TAU)
               for p in CONTROL},
        other_d025_H={p: dict(acc_v2=rows[p]["acc"], acc_d025=d25acc[p]) for p in d25["H"] if p not in CONTROL},
        spearman_65=float(rho))
    control["passed"] = all(v["below_tau"] for v in control["pairs"].values())

    # ---- H'
    H = sorted((p for p in N if rows[p]["acc"] < TAU), key=lambda p: rows[p]["acc"])
    H_new = [p for p in H if rows[p]["candidate_members"]]
    lin_new = sorted({l for p in H_new for l in rows[p]["lineages"]})
    M_H = sorted({m for p in H for m in p.split(" | ")})
    F_H = set(M_H)
    for p in N:
        a, b = p.split(" | ")
        if a in M_H or b in M_H:
            F_H.update((a, b))
    groups = components(F_H, [p.split(" | ") for p in N])
    if not control["passed"]:
        label = "NOT GIVEN (instrument control failed; Call)"
    elif len(H_new) >= 3 and len(lin_new) >= 2:
        label = "HARDER"
    elif len(H_new) == 0:
        label = "NOT HARDER"
    else:
        label = "MARGINAL"

    # ---- (a) LLMmap headline on S_val
    rec = np.mean([[float((lg[y == i].argmax(1) == i).mean()) for i in range(nm)] for lg in seeds], 0)
    top1 = [float((lg.argmax(1) == y).mean()) for lg in seeds]
    by_lin = {}
    for m, r in zip(models, rec):
        by_lin.setdefault(meta[m]["lineage"], []).append(float(r))
    headline = dict(mean_top1=float(np.mean(top1)), mean_top1_seeds=top1,
                    recall_v1=float(np.mean([r for m, r in zip(models, rec) if m in v1])),
                    recall_candidates=float(np.mean([r for m, r in zip(models, rec) if m not in v1])),
                    per_model_recall={m: round(float(r), 4) for m, r in zip(models, rec)},
                    per_lineage_recall={k: round(float(np.mean(v)), 4) for k, v in sorted(by_lin.items())},
                    linear_mean_top1=float((lin.argmax(1) == y).mean()))

    # ---- (c) population check: non-N' pairs below tau
    Nset = set(N)
    nonN = []
    for a, b in itertools.combinations(models, 2):
        p = f"{a} | {b}"
        if p in Nset:
            continue
        acc = float(np.mean([two_logit_counts(lg, y, cfg, ix[a], ix[b]).sum() / 50.0 for lg in seeds]))
        if acc < TAU:
            nonN.append(dict(pair=p, acc=acc, lineages=sorted({meta[a]["lineage"], meta[b]["lineage"]})))
    nonN.sort(key=lambda r: r["acc"])

    # ---- (d) 1-NN (D008 instrument: per-query squared distance, w2-normalised sum)
    def cube(pool):
        X = {}
        for m in models:
            s = m.replace("/", "__")
            a = np.load(f"{SCR}/embeddings/{s}.npy").astype(np.float32)
            rr = json.load(open(f"{SCR}/embeddings/{s}.index.json"))["rows"]
            sel = [(i, r["query_index"], r["config"]) for i, r in enumerate(rr) if r["pool"] == pool]
            cf = sorted({c for _, _, c in sel}); ci = {c: j for j, c in enumerate(cf)}
            cb = np.empty((8, len(cf), a.shape[1]), np.float32)
            for i, q, c in sel:
                cb[q, ci[c]] = a[i]
            X[m] = cb
        return X
    R, E = cube("build"), cube("val")
    y_ref, y_ev = np.repeat(np.arange(nm), 75), np.repeat(np.arange(nm), 25)
    Dq = np.empty((8, nm * 25, nm * 75), np.float32)
    w2 = np.empty(8)
    for q in range(8):
        A = np.concatenate([E[m][q] for m in models]); B = np.concatenate([R[m][q] for m in models])
        d = np.einsum("ij,ij->i", A, A)[:, None] + np.einsum("ij,ij->i", B, B)[None, :] - 2 * A @ B.T
        Dq[q] = np.maximum(d, 0)
        within = []
        for k in range(nm):
            Cc = B[k * 75:(k + 1) * 75]
            dd = np.maximum(np.einsum("ij,ij->i", Cc, Cc)[:, None] + np.einsum("ij,ij->i", Cc, Cc)[None, :]
                            - 2 * Cc @ Cc.T, 0)
            within.append(dd.sum() / (75 * 74))
        w2[q] = np.mean(within)
    D = summed(Dq, w2, list(range(8)))
    for p in N:
        a, b = (ix[m] for m in p.split(" | "))
        ok, _ = two_way_correct(D, y_ev, y_ref, a, b)
        rows[p]["nn1"] = float(ok.mean())
    H1 = {p for p in N if rows[p]["nn1"] < TAU}
    nn = dict(H_1NN=sorted(H1), overlap=len(H1 & set(H)), only_trained=sorted(set(H) - H1),
              only_1NN=sorted(H1 - set(H)),
              spearman_N=float(spearmanr([rows[p]["acc"] for p in N], [rows[p]["nn1"] for p in N]).correlation))
    Hlin = {p for p in N if rows[p]["linear"] < TAU}
    linear = dict(H_linear=sorted(Hlin), overlap=len(Hlin & set(H)),
                  spearman_N=float(spearmanr([rows[p]["acc"] for p in N], [rows[p]["linear"] for p in N]).correlation))

    # ---- (f) environment drift (v1 models)
    drift = {}
    for m in v1:
        s = m.replace("/", "__")
        f1 = man[m].get("f1") or {}
        dd = dict(gen_env=env[m], f1_identical_rate=f1.get("identical_rate"),
                  tokparity=tokp.get(s, {}), compat_shims=man[m].get("compat_shims"))
        try:
            new = np.load(f"{SCR}/embeddings/f1_{s}.npy").astype(np.float32)
            idx = json.load(open(f"{SCR}/embeddings/f1_{s}.index.json"))["rows"]
            old_all = np.load(f"./data/corpus_v1/embeddings/{s}.npy", mmap_mode="r")
            orow = {(r["pool"], r["config"], r["query_index"]): i for i, r in enumerate(
                json.load(open(f"./data/corpus_v1/embeddings/{s}.index.json"))["rows"])}
            old = np.stack([old_all[orow[("build", r["config"], r["query_index"])]] for r in idx]).astype(np.float32)
            Z = np.vstack([old, new]); lab = np.r_[np.zeros(len(old)), np.ones(len(new))]
            G = np.maximum(np.einsum("ij,ij->i", Z, Z)[:, None] + np.einsum("ij,ij->i", Z, Z)[None, :] - 2 * Z @ Z.T, 0)
            np.fill_diagonal(G, np.inf)
            mn = G.min(1)
            score = []
            for i in range(len(Z)):
                nbr = np.where(G[i] == mn[i])[0]
                score.append(float(np.mean(lab[nbr] == lab[i])))     # ties split
            dd["env_1nn_acc"] = float(np.mean(score))
        except FileNotFoundError:
            dd["env_1nn_acc"] = None
        drift[m] = dd

    table = sorted(rows.values(), key=lambda r: r["acc"])
    hset = dict(schema="d027-hard-set-v2", tau=TAU, boot=dict(n=NB, seed=BOOT_SEED),
                definition="DECISIONS.md D7 on the v2 universe (D027 ## D)",
                H=H, H_new=H_new, H_new_lineages=lin_new, M_H=M_H, F_H=sorted(F_H), family_groups=groups,
                H_stably_hard=[p for p in H if rows[p]["band"] == "stably_hard"],
                H_new_flags={p: {k: rows[p][k] for k in ("acc", "ci", "band", "mixed_env", "batch1",
                                                          "batch1_members", "f1_identical_rate", "tokparity",
                                                          "gen_env")} for p in H_new},
                mixed_env_pairs=[p for p in N if rows[p]["mixed_env"]],
                batch1_pairs=[p for p in N if rows[p]["batch1"]])
    json.dump(hset, open(f"{RES}/hard_set_v2.json", "w"), indent=1)
    json.dump(dict(schema="d027-universe-v2", n=nm, models=models, v1=v1,
                   candidates=[m for m in models if m not in v1],
                   gen_env=env, batch1=b1, n_prime=len(N)),
              open(f"{RES}/universe_v2.json", "w"), indent=1)
    out = dict(schema="d027-analysis-v1", label=label, n_H=len(H), n_H_new=len(H_new),
               instrument_control=control, headline=headline, pair_table=table,
               population_check_nonN_below_tau=nonN, one_nn=nn, linear=linear, env_drift=drift,
               b0_runs=runs["runs"], linear_fit=runs["linear"], wall_s=round(time.time() - t0, 1))
    json.dump(out, open(f"{RES}/analysis.json", "w"), indent=1)
    print(f"[S3] control passed={control['passed']} rho65={rho:.3f} {control['pairs']}")
    print(f"[S3] |H'|={len(H)} |H'_new|={len(H_new)} lineages={lin_new} -> LABEL {label}")
    for r in table[:25]:
        print(f"  {r['acc']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] {r['band']:11s} lin {r['linear']:.2f} "
              f"1nn {r['nn1']:.2f} {'NEW ' if r['candidate_members'] else '    '}"
              f"{'MIX ' if r['mixed_env'] else ''}{'B1 ' if r['batch1'] else ''}{r['pair']}")
    print(f"[S3] headline {headline['mean_top1']:.4f}; v1 {headline['recall_v1']:.4f} cand "
          f"{headline['recall_candidates']:.4f}; nonN<tau {len(nonN)}; wall {out['wall_s']} s")


if __name__ == "__main__":
    main()
