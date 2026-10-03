"""D028 / S0 step 4b — concurrent batch-1 processes on one GPU.

Batch-1 generation of the D027 hybrids is launch/CPU-bound (0.07-0.11 gen/s on
the pool), leaving the GH200 mostly idle. k independent processes, each
generating ONE prompt per call, keep batch-1 semantics exactly (no padding, no
batching), so no parity rule is needed; the test only measures throughput and
confirms determinism: process 0's outputs at k>1 must equal the k=1 run.

For each model and k in the list: every process loads the model, waits at a
barrier, then generates N greedy prompts (pool queries under one of the model's
greedy build configs, process-specific slice). Aggregate gen/s = k*N / wall.
-> results/D028/s0_concurrency.json

Usage:  PYTHONPATH=.:experiments <v2 python> experiments/d028_s0_concurrency.py
"""
import os
import json
import time
import multiprocessing as mp

N = 12
PLAN = {"tiiuae/Falcon-H1-3B-Instruct": [1, 4, 8], "tiiuae/Falcon-H1-7B-Instruct": [1, 4],
        "Qwen/Qwen3.5-4B": [1, 4, 8]}


def worker(m, k, rank, barrier, q):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch
    import d006_s4_shard as h
    from LLMmap.prompt_configuration import PromptConf
    from d027_llm import LLMv2
    from d028_s0 import revision_and_build
    rev, build = revision_and_build(m)
    llm = LLMv2(m, rev)
    gi = [i for i, c in enumerate(build) if not c["sampling_hparams"]["do_sample"]][0]
    conf = PromptConf.from_dict(dict(build[gi], sampling_hparams=dict(do_sample=False)))
    queries = [e["text"] for e in json.load(open(h.Q0))["queries"]]
    prompts = [conf(qq, llm)[0] for qq in queries[rank * N:(rank + 1) * N]]
    barrier.wait()
    t = time.time()
    outs = [llm.generate([p], dict(do_sample=False), max_new_tokens=h.TOKEN_CEILING)[0] for p in prompts]
    q.put(dict(rank=rank, wall=time.time() - t, outs=outs if rank == 0 else None,
               mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)))


def main():
    ctx = mp.get_context("spawn")
    res = {}
    for m, ks in PLAN.items():
        res[m] = {}
        ref = None
        for k in ks:
            barrier, q = ctx.Barrier(k), ctx.Queue()
            ps = [ctx.Process(target=worker, args=(m, k, r, barrier, q)) for r in range(k)]
            for p in ps:
                p.start()
            got = [q.get() for _ in range(k)]
            for p in ps:
                p.join()
            wall = max(g["wall"] for g in got)
            r0 = next(g for g in got if g["rank"] == 0)
            if k == 1:
                ref = r0["outs"]
            res[m][k] = dict(aggregate_gen_per_s=round(k * N / wall, 3), wall_s=round(wall, 1),
                             per_process_mem_gb=max(g["mem_gb"] for g in got),
                             rank0_identical_to_k1=(r0["outs"] == ref))
            print(m, k, res[m][k], flush=True)
            json.dump(res, open("./results/D028/s0_concurrency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
