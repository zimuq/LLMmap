"""D030 / S1 — job layout. Writes results/D030/lists/bucket_NN.txt (one model per line) and
results/D030/lists/layout.json. Padded models are packed (LPT) into buckets of ~TARGET_H
conservative hours (s0.json projection `min|25`); the batch-1 trio gets one job each with D028's
K. Wall per bucket = 1.6 x conservative + 1 h, capped at 47 h.
"""
import os
import json

TARGET_H = 10.0
TRIO_K = {"tiiuae/Falcon-H1-3B-Instruct": 8, "tiiuae/Falcon-H1-7B-Instruct": 4, "Qwen/Qwen3.5-4B": 8}
DBLBOS = ("meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
          "google/gemma-2-9b-it", "meta-llama/Llama-2-7b-chat-hf")
OUT = "./results/D030/lists"


def main():
    s0 = json.load(open("./results/D030/s0.json"))
    cons = s0["projection"]["by_T_ZP"]["min|25"]["per_model"]
    cent = s0["projection"]["by_T_ZP"]["median|25"]["per_model"]
    padded = sorted((m for m in cons if m not in TRIO_K), key=lambda m: -cons[m]["total_h"])
    nb = max(1, round(sum(cons[m]["total_h"] for m in padded) / TARGET_H))
    buckets = [[] for _ in range(nb)]
    load = [0.0] * nb
    for m in padded:
        i = load.index(min(load))
        buckets[i].append(m)
        load[i] += cons[m]["total_h"] + (0.1 if m in DBLBOS else 0)
    os.makedirs(OUT, exist_ok=True)
    layout = []
    for i, b in enumerate(buckets):
        f = f"{OUT}/bucket_{i:02d}.txt"
        open(f, "w").write("\n".join(b) + "\n")
        wall = min(47.0, 1.6 * load[i] + 1.0)
        layout.append(dict(job=f"b{i:02d}", list=f, models=b, K=1, conservative_h=round(load[i], 2),
                           central_h=round(sum(cent[m]["total_h"] for m in b), 2), wall_h=round(wall, 1)))
    for m, k in TRIO_K.items():
        layout.append(dict(job="trio-" + m.split("/")[1][:12], model=m, K=k,
                           conservative_h=cons[m]["total_h"], central_h=cent[m]["total_h"],
                           wall_h=round(min(47.0, 1.6 * cons[m]["total_h"] + 1.0), 1)))
    json.dump(dict(target_h=TARGET_H, n_jobs=len(layout), layout=layout), open(f"{OUT}/layout.json", "w"), indent=1)
    for x in layout:
        print(x["job"], x.get("K"), x["central_h"], x["conservative_h"], x["wall_h"], len(x.get("models", [1])))


if __name__ == "__main__":
    main()
