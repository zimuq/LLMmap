"""D027 / S0 step 3 — stage candidate weights to $SCRATCH/hf-cache from a LOGIN
node (ENV.md: compute-node egress untested). Revisions are pinned to the SHAs
recorded in results/D027/hf_prefacts.json (2026-10-01), so a moved `main`
cannot change weights or templates under us.

Excluded: Mistral's duplicate `consolidated*` checkpoint, `original/`, and
non-safetensors weight formats (every candidate ships safetensors).
Mistral-Small-3.2 is staged WITHOUT weights: it ships no tokenizer config or
HF chat template (rule 6), so S0 only needs its small files to re-confirm the
drop.

Sequential and restartable (snapshot_download skips complete files).
Writes results/D027/staging.json.

Usage:  HF_HOME=$SCRATCH/hf-cache python experiments/d027_stage.py
"""
import os
import json
import time

from huggingface_hub import snapshot_download

PRE = "./results/D027/hf_prefacts.json"
OUT = "./results/D027/staging.json"
IGNORE = ["consolidated*", "original/*", "*.pth", "*.pt", "*.bin", "*.gguf",
          "*.onnx", "onnx/*", "*.msgpack", "*.h5"]
NO_WEIGHTS = {"mistralai/Mistral-Small-3.2-24B-Instruct-2506"}


def main():
    tok = open("/work/11280/zimuq1/vista/.hf_token").read().strip()
    pre = json.load(open(PRE))
    log = json.load(open(OUT)) if os.path.exists(OUT) else {}
    order = sorted(pre, key=lambda m: pre[m].get("weights_gb") or 0)   # small first
    for m in order:
        if log.get(m, {}).get("status") == "OK":
            continue
        ign = IGNORE + (["*.safetensors"] if m in NO_WEIGHTS else [])
        t = time.time()
        try:
            p = snapshot_download(m, revision=pre[m]["sha"], token=tok,
                                  ignore_patterns=ign, max_workers=8)
            gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p, followlinks=True)
                     for f in fs) / 1e9
            log[m] = dict(status="OK", revision=pre[m]["sha"], path=p, gb=round(gb, 1),
                          wall_s=round(time.time() - t, 1), weights=m not in NO_WEIGHTS)
        except Exception as e:
            log[m] = dict(status="FAILED", revision=pre[m]["sha"], error=repr(e)[:500],
                          wall_s=round(time.time() - t, 1))
        print(m, log[m]["status"], log[m].get("gb"), log[m]["wall_s"], flush=True)
        json.dump(log, open(OUT, "w"), indent=1)
    ok = sum(v["status"] == "OK" for v in log.values())
    print(f"staged {ok}/{len(pre)}; total {sum(v.get('gb', 0) for v in log.values()):.0f} GB")


if __name__ == "__main__":
    main()
