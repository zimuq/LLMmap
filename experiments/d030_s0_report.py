"""D030 / S0 — assemble results/D030/s0.json: freeze, checks, pilot, projection, precision.

Any python (CPU). Inputs: s0_freeze.json, s0_checks_*.json, results/D030/s0_pilot/*.status.json,
pilot slurm logs (node wall), results/D028/manifest.json (multilingual-extension gen/s per model,
same env / mode / revision as baseline generation), results/D029/analysis.json.

Projection rule (P1 S0.6):
  padded models : rate(m, method, row) = D028 ML gen/s(m) x f(method, row), f = median over the
                  padded pilot models of pilot gen/s / D028 ML gen/s (min / max reported).
  batch-1 trio  : their own pilot, node rate = K x median per-process gen/s over the parts that
                  ran that row (processes run concurrently on one GPU, D028 Call 1).
  per job       : + measured load overhead (max over single-process pilots), 15-min floor.
  makespan      : one job per model (trio: as D028, 1-2 jobs), LPT on 20 slots (MaxJobsPU).
Precision (D's rule): half-width x sqrt(25 / T_ZP) from D029's primary intervals.
"""
import os
import re
import glob
import json
import math
import hashlib
import statistics as st

OUT = "./results/D030"
PILOT = f"{OUT}/s0_pilot"
TRIO_K = {"tiiuae/Falcon-H1-3B-Instruct": 8, "tiiuae/Falcon-H1-7B-Instruct": 4, "Qwen/Qwen3.5-4B": 8}
ROWTYPE = {"gen_ref": "native", "native_tgt": "native", "all_ref": "all", "test_tgt": "test",
           "test_timing": "test"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def node_walls():
    w = {}
    for f in glob.glob(f"{OUT}/logs/pilot-*.out"):
        if re.search(r"pilot-\d+-p\d+\.log$", f):
            continue
        txt = open(f).read()
        m = re.search(r"model: (\S+) K=(\d+)", txt)
        n = re.search(r"node_wall_s (\d+)", txt)
        if m and n:
            w[m.group(1)] = dict(K=int(m.group(2)), node_wall_s=int(n.group(1)), log=os.path.basename(f))
    return w


def pilot_rates():
    per = {}
    for f in glob.glob(f"{PILOT}/*.status.json"):
        s = json.load(open(f))
        per.setdefault(s["model"], []).append(s)
    out = {}
    for m, ss in per.items():
        r = dict(parts=len(ss), peak_mem_gb=max(s.get("peak_mem_gb", 0) for s in ss), rows={})
        for meth in ("met", "zp"):
            for row in ("gen_ref", "all_ref", "test_timing"):
                rates, n, w, tok, ln, em, bs = [], 0, 0.0, 0, 0, 0, set()
                for s in ss:
                    b = s["blocks"].get(meth, {}).get("by_row", {}).get(row)
                    if not b or b["wall_s"] <= 0:
                        continue
                    rates.append(b["n_gen"] / b["wall_s"])
                    n += b["n_gen"]; w += b["wall_s"]; tok += b["n_tok"]; ln += b["n_length"]
                    em += b["n_empty"]; bs.update(b["batches"])
                if not rates:
                    continue
                K = TRIO_K.get(m, 1)
                node_rate = K * st.median(rates) if K > 1 else n / w
                r["rows"][f"{meth}|{row}"] = dict(n_gen=n, wall_s=round(w, 1), node_gen_per_s=round(node_rate, 3),
                                                  mean_new_tokens=round(tok / n, 1),
                                                  cap_hit_rate=round(ln / n, 3), empty_rate=round(em / n, 4),
                                                  batches=sorted(bs))
        r["gen_wall_total_s"] = round(sum(sum(b["wall_s"] for b in s["blocks"].get(k, {}).get("by_row", {}).values())
                                          for s in ss for k in ("met", "zp")), 1)
        out[m] = r
    return out


def ml_rate(man, m):
    ps = man["models"][m]["ml_final"]["part_status"]
    if m in TRIO_K:      # concurrent parts: node rate = sum of per-part rates
        return sum(p["gen_per_s"] for p in ps)
    return ps[0]["gen_per_s"]


def lpt(durations, slots=20):
    loads = [0.0] * slots
    for d in sorted(durations, reverse=True):
        i = loads.index(min(loads)); loads[i] += d
    return max(loads)


def main():
    freeze = json.load(open(f"{OUT}/s0_freeze.json"))
    manifest = json.load(open("./confs/baselines/baseline_ext_v1.json"))
    d028 = json.load(open("./results/D028/manifest.json"))
    checks = {os.path.basename(f)[10:-5]: json.load(open(f)) for f in glob.glob(f"{OUT}/s0_checks_*.json")}
    pilot, walls = pilot_rates(), node_walls()

    # load overhead from single-process pilots
    over = [walls[m]["node_wall_s"] - pilot[m]["gen_wall_total_s"] for m in pilot
            if m not in TRIO_K and m in walls]
    overhead_s = max(over) if over else 600.0
    # calibration factors on padded pilots
    padded = [m for m in pilot if m not in TRIO_K]
    fac = {}
    for key in {k for m in padded for k in pilot[m]["rows"]}:
        rs = [pilot[m]["rows"][key]["node_gen_per_s"] / ml_rate(d028, m) for m in padded if key in pilot[m]["rows"]]
        fac[key] = dict(median=round(st.median(rs), 4), min=round(min(rs), 4), max=round(max(rs), 4), n=len(rs))

    def rate(m, meth, rt):
        key = f"{meth}|{dict(native='gen_ref', all='all_ref', test='test_timing')[rt]}"
        if m in TRIO_K:
            return pilot[m]["rows"][key]["node_gen_per_s"]
        return ml_rate(d028, m) * fac[key]["median"]

    vol = dict(met=dict(native=500, all=250, test_per_slot=250), zp=dict(native=400, all=200, test_per_slot=200))
    proj = {}
    for T in (5, 10, 25):
        per_model, jobs = {}, []
        for m in manifest["models"]:
            s_met = vol["met"]["native"] / rate(m, "met", "native") + vol["met"]["all"] / rate(m, "met", "all") \
                + 25 * vol["met"]["test_per_slot"] / rate(m, "met", "test")
            s_zp = vol["zp"]["native"] / rate(m, "zp", "native") + vol["zp"]["all"] / rate(m, "zp", "all") \
                + T * vol["zp"]["test_per_slot"] / rate(m, "zp", "test")
            h = (s_met + s_zp + overhead_s) / 3600
            nj = max(1, math.ceil(h / 40))                 # keep each job well under the 48 h wall
            per_model[m] = dict(met_h=round(s_met / 3600, 2), zp_h=round(s_zp / 3600, 2), total_h=round(h, 2), jobs=nj)
            jobs += [max(0.25, h / nj)] * nj
        tot = sum(max(0.25, v["total_h"]) for v in per_model.values())
        proj[str(T)] = dict(node_h_total=round(tot, 1),
                            node_h_met=round(sum(v["met_h"] for v in per_model.values()), 1),
                            node_h_zp=round(sum(v["zp_h"] for v in per_model.values()), 1),
                            node_h_trio=round(sum(per_model[m]["total_h"] for m in TRIO_K), 1),
                            makespan_h_20_slots=round(lpt(jobs), 1), n_jobs=len(jobs),
                            max_model_h=max(per_model.items(), key=lambda kv: kv[1]["total_h"]),
                            per_model=per_model)

    a = json.load(open("./results/D029/analysis.json"))["primary"]
    hw_all = (a["E_all"]["ci"][1] - a["E_all"]["ci"][0]) / 2
    hw_hard = (a["E_hard"]["ci"][1] - a["E_hard"]["ci"][0]) / 2
    prec = {str(T): dict(bootstrap_clusters_zp=T, clusters_other_arms=25, traces_per_pair=2 * T,
                         halfwidth_delta_all=round(hw_all * math.sqrt(25 / T), 4),
                         halfwidth_delta_hard=round(hw_hard * math.sqrt(25 / T), 4),
                         parity_reachable_all=hw_all * math.sqrt(25 / T) < 0.02,
                         parity_reachable_hard=hw_hard * math.sqrt(25 / T) < 0.02)
            for T in (5, 10, 25)}

    s0 = dict(
        schema="d030-s0-v1",
        repos=dict(zeroprint=manifest["protocol"]["zp"]["repo_commit"], met=manifest["protocol"]["met"]["repo_commit"]),
        prompts=manifest["prompts"], manifest_sha256=sha("./confs/baselines/baseline_ext_v1.json"),
        freeze=freeze, checks=checks,
        pilot=dict(rates=pilot, node_walls=walls, load_overhead_s=overhead_s, calibration=fac,
                   note="pilot rows never part of baseline_ext_v1; test_timing rows timing-only (Review A3)"),
        projection=dict(rule=__doc__.split("Projection rule (P1 S0.6):")[1].split("Precision")[0].strip(),
                        by_T_ZP=proj),
        precision=dict(d029_primary=dict(E_all_ci=a["E_all"]["ci"], E_hard_ci=a["E_hard"]["ci"],
                                         halfwidth_all=round(hw_all, 4), halfwidth_hard=round(hw_hard, 4)),
                       by_T_ZP=prec))
    json.dump(s0, open(f"{OUT}/s0.json", "w"), indent=1, ensure_ascii=False)
    print("calibration:", json.dumps(fac, indent=1))
    for m, r in pilot.items():
        print(m, r["parts"], r["peak_mem_gb"], {k: (v["node_gen_per_s"], v["mean_new_tokens"], v["cap_hit_rate"])
                                                 for k, v in r["rows"].items()})
    for T, p in proj.items():
        print("T_ZP", T, {k: v for k, v in p.items() if k != "per_model"})
    print(json.dumps(prec, indent=1))


if __name__ == "__main__":
    main()
