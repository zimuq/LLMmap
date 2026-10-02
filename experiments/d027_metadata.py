"""D027 / S0 step 7 — candidate metadata (verbatim from the D's frozen table)
and N' (the unchanged near_relative_pairs rule over 37 + candidates).

Columns follow results/D001/model_metadata.csv. Only `lineage` and `base`
enter the N rule; they are copied verbatim from `## D`. Other columns are
filled from the D (P -> params_b) and from hf_prefacts.json (model_type ->
arch); `tok` is left empty (no candidate tokenizer-family audit was
pre-registered, and nothing reads it for N).

Usage:  python experiments/d027_metadata.py [--survivors results/D027/survivors.txt]
"""
import re
import csv
import json
import argparse
import itertools

COLS = ["model", "org", "lineage", "variant", "base", "base_prov", "arch", "tok",
        "params_b", "active_b", "proprietary"]
OUT_CSV = "./results/D027/candidates_metadata.csv"


def d_table():
    rows = []
    for l in open("./docs/D027.md"):
        if re.match(r"^\| \d+ \| ", l):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            rows.append(dict(n=int(c[0]), model=c[1], lineage=c[2], base=c[3], P=c[4], date=c[5]))
    assert len(rows) == 49
    return rows


def pairs(rows, models):
    r = {x["model"]: x for x in rows}
    out = []
    for a, b in itertools.combinations(models, 2):
        sb = bool(r[a]["base"]) and r[a]["base"] == r[b]["base"]
        sl = bool(r[a]["lineage"]) and r[a]["lineage"] == r[b]["lineage"]
        if sb or sl:
            out.append(dict(pair=f"{a} | {b}", same_base=sb, same_lineage=sl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors")
    a = ap.parse_args()
    pre = json.load(open("./results/D027/hf_prefacts.json"))
    cand = []
    for r in d_table():
        org, name = r["model"].split("/")
        cand.append(dict(model=r["model"], org=org.lower(), lineage=r["lineage"],
                         variant=name.lower(), base=r["base"],
                         base_prov="model card (D027 table)" if r["base"] else "",
                         arch=pre[r["model"]].get("model_type", ""), tok="",
                         params_b=r["P"], active_b="", proprietary="False"))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(cand)

    v1_models = sorted(s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"])
    v1_rows = [r for r in csv.DictReader(open("./results/D001/model_metadata.csv")) if r["model"] in v1_models]
    assert len(v1_rows) == 37
    allrows = v1_rows + cand
    n_v1 = pairs(allrows, v1_models)
    d25 = json.load(open("./results/D025/hard_set.json"))
    assert len(n_v1) == 65, len(n_v1)
    surv = [l.strip() for l in open(a.survivors)] if a.survivors else [c["model"] for c in cand]
    universe = sorted(v1_models + surv)
    N = pairs(allrows, universe)
    # rule 5 check: every survivor has at least one N' neighbour
    lonely = [m for m in surv if not any(m in p["pair"].split(" | ") for p in N)]
    out = dict(n_v1_pairs=len(n_v1), n_universe=len(universe), n_pairs=len(N),
               n_pairs_with_candidate=sum(any(m in surv for m in p["pair"].split(" | ")) for p in N),
               candidates_without_neighbour=lonely, survivors_used=bool(a.survivors), pairs=N)
    json.dump(out, open("./results/D027/n_prime.json", "w"), indent=1)
    print(f"N restricted to v1 = {len(n_v1)} (== 65 asserted); universe {len(universe)}; "
          f"|N'| = {len(N)} ({out['n_pairs_with_candidate']} with a candidate); "
          f"without neighbour: {lonely}")


if __name__ == "__main__":
    main()
