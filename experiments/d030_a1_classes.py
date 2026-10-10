"""D030 / S4 — per-model Call A-1 class per method (design-side taxonomy, Review stop A), derived from
the S0.5(c) parity records (results/D030/s0_checks_parity_*.json). No score is read.
  equal  our rendered native prompt tokenises identically to the official path
  A      equal once template kwargs are removed (thinking off / frozen date)
  D      the official path adds a BOS that our template (and corpus) never had (Issue 24)
  B      ZP only: official double BOS, nothing else wrong (inside / like ZeroPrint's published scope)
  C      anything else (MET BOS-string removal deleting a template token, merged turns, appended EOS;
         ZP 512-token native-prompt truncation, aya-23)
Writes results/D030/a1_classes.json.
"""
import glob
import json


def classify(meth, r, bos):
    x = r[meth]
    if x["equal"]:
        return "equal"
    if x["equal_without_template_kwargs"]:
        return "A"
    ex = x["examples"][0]
    o, f = ex["ours_head"], ex["official_head"]
    if o[0] != bos and f[0] == bos and f[1:] == o[:len(f) - 1] and ex["n_official"] == ex["n_ours"] + 1:
        return "D"
    if meth == "zp" and x.get("official_double_bos") and x["equal_modulo_bos"]:
        return "B"
    return "C"


def main():
    rec = {}
    for f in sorted(glob.glob("./results/D030/s0_checks_parity_*.json")):
        for m, r in json.load(open(f)).items():
            if isinstance(r, dict) and "met" in r:
                rec[m] = r
    assert len(rec) == 85, len(rec)
    out = {}
    for m, r in sorted(rec.items()):
        out[m] = dict(met=classify("met", r, r["bos"]), zp=classify("zp", r, r["bos"]),
                      zp_double_bos=bool(r["zp"].get("official_double_bos")),
                      zp_test_slots_truncated=len(r.get("zp_trunc", {}).get("test_slots_affected", [])))
    cnt = {k: {c: sum(1 for v in out.values() if v[k] == c) for c in ("equal", "A", "B", "C", "D")} for k in ("met", "zp")}
    json.dump(dict(counts=cnt, models=out), open("./results/D030/a1_classes.json", "w"), indent=1)
    print(cnt)


if __name__ == "__main__":
    main()
