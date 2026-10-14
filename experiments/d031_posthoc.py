"""D031 / S6 — post-hoc paired contrasts (Review D031/R, 2026-10-14). Descriptive; not pre-registered.

Contrasts at k = 8, both views x both readouts x {E_all, all-pair E_tail}:
  MMR* - PAPER8, TOP-K - PAPER8, MMR* - TOP-K, JG - PAPER8 (reference row).
MMR* = the S_val-starred variant in variant_choice.json (MMR|pearson|0.7 in both views); no other variant.
Stored counts only (S2 counts for TOP-K / MMR*; JG / PAPER8 as S4 loads them); no refits.
Bootstrap: S4's Boot (same replicate indices), so every interval is jointly drawn with the recorded ones.
Asserts: every level equals analysis.json -> curves at k = 8; JG - TOP-K and JG - MMR* intervals equal the S4
labels exactly (same replicates); JG - PAPER8 equals D029 primary (v2 attention E_all) and D023 S_b_tail_k8
(v1 attention E_tail) points.
Writes results/D031/posthoc_contrasts.json.
Usage:  PYTHONPATH=.:experiments python experiments/d031_posthoc.py
"""
import json
import time

import d031_lib as L
from d031_analysis import Boot, Store, Arm, contrast, label, r4, RES

PAIRS = (("MMR*", "PAPER8"), ("TOP-K", "PAPER8"), ("MMR*", "TOP-K"), ("JG", "PAPER8"))


def main():
    t0 = time.time()
    s4 = json.load(open(f"{RES}/analysis.json"))
    vc = json.load(open(f"{RES}/variant_choice.json"))
    out = dict(schema="d031-posthoc-v1", tag="post-hoc", k=8,
               note="descriptive, not pre-registered; cannot change any D031 label; same replicates as S4",
               bootstrap=s4["bootstrap"], views={})
    for view in ("v2", "v1"):
        B, st = Boot(view), Store(view)
        mmr = vc["views"][view]["MMR_star"]
        names = {"JG": "JG", "PAPER8": "PAPER8", "TOP-K": "TOP-K", "MMR*": mmr}
        ev = {}
        for ro in ("att", "lin"):
            for a, nm in names.items():
                X = Arm(nm, st.ref(ro, nm, 8)) if nm in ("JG", "PAPER8") else Arm(nm, st.new(ro, nm, 8))
                ev[(ro, a)] = dict(all=X.e_all(B), tail=X.e_tail(B))
                c = s4["curves"][view][f"{ro}|{nm}"]["8"]
                assert r4(ev[(ro, a)]["all"][1]) == c["E_all"] and r4(ev[(ro, a)]["tail"][1]) == c["E_tail"], (view, ro, nm)
        for x, nm in (("TOP-K", "TOP-K"), ("MMR*", mmr)):                 # same replicates as S4
            lab = s4["labels"][view]["TOP-K" if x == "TOP-K" else "MMR*"]["att"]
            for e, f in (("all", "d_all"), ("tail", "d_tail")):
                got = contrast(ev[("att", "JG")][e], ev[("att", x)][e])
                assert got == lab[f], (view, x, e, got, lab[f])
        cells = {}
        for a, b in PAIRS:
            row = {}
            for ro in ("att", "lin"):
                row[ro] = {f"d_{e}": contrast(ev[(ro, a)][e], ev[(ro, b)][e]) for e in ("all", "tail")}
            lp = {ro: (row[ro]["d_all"]["delta"], row[ro]["d_tail"]["delta"]) for ro in ("att", "lin")}
            row["label_post_hoc"] = label(row["att"]["d_all"], row["att"]["d_tail"], *lp["lin"])
            row["linear_as_label_post_hoc"] = label(row["lin"]["d_all"], row["lin"]["d_tail"], *lp["lin"])  # S4 convention
            cells[f"{a} - {b}"] = row
        jp = cells["JG - PAPER8"]["att"]
        if view == "v2":
            d29 = json.load(open("./results/D029/analysis.json"))["primary"]["E_all"]
            assert jp["d_all"]["delta"] == d29["delta"], (jp["d_all"], d29)
            cells["JG - PAPER8"]["D029_primary_E_all"] = d29
        else:
            d23 = json.load(open("./results/D023/analysis.json"))["S_b_tail_k8"]
            assert jp["d_tail"]["a"] == d23["JointGreedy"]["cvar01"] and jp["d_tail"]["b"] == d23["paper8"]["cvar01"]
            cells["JG - PAPER8"]["D023_S_b_tail_k8_paper8_minus_JG"] = d23["paper8"]["minus_JG"]
        out["views"][view] = dict(MMR_star=mmr, cells=cells)
        for k, r in cells.items():
            print(f"[S6 {view}] {k:16s} att all {r['att']['d_all']['delta']:+.4f} {r['att']['d_all']['ci']} "
                  f"tail {r['att']['d_tail']['delta']:+.4f} {r['att']['d_tail']['ci']} -> {r['label_post_hoc']} "
                  f"(lin-as-label {r['linear_as_label_post_hoc']})", flush=True)
    out["wall_s"] = round(time.time() - t0, 1)
    L.jdump(out, f"{RES}/posthoc_contrasts.json")
    print(f"[S6] wrote posthoc_contrasts.json ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
