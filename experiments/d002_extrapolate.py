"""
D002 / P1 — S3 (M4): extrapolate Phase-1 cost from MEASURED throughput.

Cost model, stated explicitly so a reviewer can substitute their own assumptions:

    generations  = |L| x |Q_0| x n_configs
    node_hours   = generations / gen_per_s / 3600  +  |L| x load_s / 3600

Rates come from results/D002/throughput_benchmark.csv (measured, not estimated).
The two benchmark models are used as a BRACKET rather than a fitted scaling law --
see R: throughput did not track parameter count (the 28-layer 7B beat the 36-layer
3B at batch 1), so size-based interpolation is not defensible.

Vista charge rate: gh = 1 SU per node-hour (min 15 min charged per job).
"""
import csv
import json
import itertools

BENCH = "./results/D002/throughput_benchmark.csv"
OUT = "./results/D002/cost_extrapolation.csv"
SU_PER_NODE_HOUR = 1.0          # gh queue
AVAIL_SU = 6982                 # TG-NAIRR250513 balance at time of measurement
N_CONFIGS = 125                 # C4 proposal (75/25/25) -- itself undecided
LOAD_S = 30.0                   # cold model load, from measured 13.3s (3B) / 29.3s (7B)

CANDIDATES = [(20, 80), (25, 100), (30, 120)]


def main():
    rows = list(csv.DictReader(open(BENCH)))
    # rate[(regime, max_new_tokens)] = (min gen/s, max gen/s) across benchmark models
    rate = {}
    for r in rows:
        k = (r["regime"], int(r["max_new_tokens"]))
        g = float(r["gen_per_s"])
        lo, hi = rate.get(k, (g, g))
        rate[k] = (min(lo, g), max(hi, g))

    out = []
    for (L, Q0), (regime, mnt) in itertools.product(CANDIDATES, sorted(rate)):
        gens = L * Q0 * N_CONFIGS
        lo, hi = rate[(regime, mnt)]
        load_h = L * LOAD_S / 3600.0
        # fast rate -> low cost, slow rate -> high cost
        h_lo = gens / hi / 3600.0 + load_h
        h_hi = gens / lo / 3600.0 + load_h
        out.append(dict(
            L=L, Q0=Q0, n_configs=N_CONFIGS, generations=gens,
            regime=regime, max_new_tokens=mnt,
            gen_per_s_lo=lo, gen_per_s_hi=hi,
            node_hours_lo=round(h_lo, 1), node_hours_hi=round(h_hi, 1),
            SU_lo=round(h_lo * SU_PER_NODE_HOUR, 1),
            SU_hi=round(h_hi * SU_PER_NODE_HOUR, 1),
            pct_of_balance_hi=round(h_hi * SU_PER_NODE_HOUR / AVAIL_SU * 100, 1),
            fits_in_balance=bool(h_hi * SU_PER_NODE_HOUR <= AVAIL_SU),
            wall_days_hi_single_job=round(h_hi / 24.0, 1),
        ))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    hdr = f"{'|L|x|Q0|':>10} {'regime':>10} {'tok':>4} {'gens':>8} {'node-h':>14} {'%bal':>6} {'days':>6}"
    print(hdr); print("-" * len(hdr))
    for r in out:
        print(f"{str(r['L'])+'x'+str(r['Q0']):>10} {r['regime']:>10} {r['max_new_tokens']:>4} "
              f"{r['generations']:>8} {r['node_hours_lo']:>6.1f}-{r['node_hours_hi']:<7.1f} "
              f"{r['pct_of_balance_hi']:>5.1f}% {r['wall_days_hi_single_job']:>5.1f}")

    speed = {}
    for mnt in sorted({int(r["max_new_tokens"]) for r in rows}):
        b = rate[("batched", mnt)]; u = rate[("unbatched", mnt)]
        speed[mnt] = (round(b[0]/u[1], 2), round(b[1]/u[0], 2))
    print("\nbatch-8 speedup (x) by max_new_tokens:", json.dumps(speed))
    print(f"balance: {AVAIL_SU} SU @ {SU_PER_NODE_HOUR} SU/node-hour")


if __name__ == "__main__":
    main()
