"""
D002 / P1 — S1 (M1/M2 GPU verification) + S2 (M3 throughput benchmark).

Run from repo root with PYTHONPATH=. on a `gh` / `gh-dev` node.
NOTE: Vista does NOT support --gres/--gpus-per-task (Slurm rejects them); GH nodes
are exclusive so the H200 is simply present.

Measures BOTH regimes:
  unbatched  -- one prompt per generate() call, exactly what
                dataset_maker.make_dataset_entries_for_new_llm:36-39 does today
  batched    -- the 8 queries of one prompt config in a single generate() call
                (valid: identical sampling params within a config)
The ratio is the number that decides Phase-1 affordability.
"""
import os
import json
import time
import argparse

import torch

from LLMmap.llm import load_llm
from LLMmap.prompt_configuration import PromptConfFactory

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]


def verify_gpu(out_dir):
    """M1 + M2. Never infer from `pip list` -- do a real on-device op."""
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_is_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["total_mem_GB"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)

        before = torch.cuda.memory_allocated()
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()

        ref = (a.cpu() @ b.cpu())
        info["on_device_op"] = {
            "tensor_device": str(c.device),
            "is_cuda": c.device.type == "cuda",
            "mem_allocated_delta_MB": round((after - before) / 1024**2, 1),
            "matches_cpu_reference": bool(
                torch.allclose(c.cpu(), ref, atol=1e-2, rtol=1e-2)),
        }
        del a, b, c
        torch.cuda.empty_cache()

    print("[S1]", json.dumps(info, indent=2), flush=True)
    with open(f"{out_dir}/gpu_verification.json", "w") as fh:
        json.dump(info, fh, indent=2)
    return info


def bench_model(name, queries, confs, device_ok, max_new_tokens):
    """Return per-regime timing for one model."""
    rows = []
    torch.cuda.reset_peak_memory_stats() if device_ok else None

    t0 = time.time()
    llm = load_llm(name, 0)  # llm_type 0 = HuggingFace, device_map="auto"
    load_s = time.time() - t0
    dev = next(llm.model.parameters()).device
    print(f"  loaded {name} in {load_s:.0f}s on {dev}", flush=True)

    for regime in ("unbatched", "batched"):
        n_gen, n_tok, errs = 0, 0, 0
        torch.cuda.synchronize() if device_ok else None
        t0 = time.time()
        for conf in confs:
            prompts, params = [], None
            for q in queries:
                p, params = conf(q, llm)
                prompts.append(p)
            try:
                if regime == "unbatched":
                    outs = [llm.generate(p, dict(params),
                                         max_new_tokens=max_new_tokens)[0]
                            for p in prompts]
                else:
                    outs = llm.generate(prompts, dict(params),
                                        max_new_tokens=max_new_tokens)
            except Exception as e:
                errs += 1
                print(f"    [warn] {regime} conf failed: {type(e).__name__} {e}",
                      flush=True)
                continue
            n_gen += len(outs)
            n_tok += sum(len(llm.tokenizer(o, add_special_tokens=False).input_ids)
                         for o in outs)
        torch.cuda.synchronize() if device_ok else None
        wall = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1024**3) if device_ok else 0.0
        rows.append(dict(
            model=name, regime=regime, max_new_tokens=max_new_tokens,
            n_generations=n_gen, gen_tokens=n_tok, wall_s=round(wall, 2),
            gen_per_s=round(n_gen / wall, 3) if wall else 0,
            tokens_per_s=round(n_tok / wall, 1) if wall else 0,
            mean_tokens_per_gen=round(n_tok / n_gen, 1) if n_gen else 0,
            load_s=round(load_s, 1), peak_gpu_GB=round(peak, 2),
            failed_confs=errs, device=str(dev),
        ))
        print(f"  [{regime}] {n_gen} gens, {wall:.1f}s, "
              f"{rows[-1]['gen_per_s']} gen/s, {rows[-1]['tokens_per_s']} tok/s",
              flush=True)

    del llm
    if device_ok:
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./results/D002")
    ap.add_argument("--n_confs", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, nargs="+", default=[100],
                    help="C7 is currently inert; 100 is llm.py's default. Pass "
                         "'100 200' to price a longer-response setting too.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    info = verify_gpu(args.out)
    device_ok = bool(info.get("cuda_is_available"))
    if not device_ok:
        print("[S1] CUDA NOT AVAILABLE -- proceeding as the M5 (CPU) branch",
              flush=True)

    queries = json.load(open("./confs/queries/default.json"))
    pc = PromptConfFactory("./confs/prompt_configurations/")
    confs = pc.sample(args.n_confs)
    print(f"[S2] {len(queries)} queries x {len(confs)} configs "
          f"= {len(queries)*len(confs)} generations per model per regime",
          flush=True)

    all_rows = []
    for mnt in args.max_new_tokens:
        for m in MODELS:
            all_rows += bench_model(m, queries, confs, device_ok, mnt)

    import csv
    with open(f"{args.out}/throughput_benchmark.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print("[S2] wrote throughput_benchmark.csv", flush=True)
    print(json.dumps(all_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
