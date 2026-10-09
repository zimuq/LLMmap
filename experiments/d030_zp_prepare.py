"""D030 / S0.3 — freeze ZeroPrint's 10 prompts and their query embeddings.

Run with envs/d030-zp/bin/python on a compute node. Calls the OFFICIAL code path at the
pinned commit: `utils.utils.setup_environment` (seed 1000 from config/zeroprint.yaml) and
`ZeroPrintFingerprint(config).prepare()` with the shipped config unchanged except that the
cache paths are redirected into results/D030/zp_prepare/ (cwd). prepare() reservoir-samples
2 HumanEval queries, makes 4 GloVe substitutions each (thread pool on the global `random`),
and embeds all 10 with all-mpnet-base-v2. It is run ONCE and frozen (P1 Call 9; Review:
re-running would itself be a selection).

Native quirk recorded, not changed: `original_queries = list(set(...))`, so the order of the
two base queries depends on the process's string-hash seed. The frozen `all_queries` order
(originals first, then the perturbations in data order) is the order every generation and
every fingerprint uses from here on.

Outputs (results/D030/zp_prepare/): the official cache files (csv + .pth) and
confs/baselines/zp_prompts.json (+ zp_query_emb.npy).
"""
import os
import sys
import json
import hashlib
import argparse
import subprocess

ZP_REPO = "/work/11280/zimuq1/vista/ext/ZeroPrint"
ZP_COMMIT = "16a02aa4cfd5693ecfa757e9d2832b7e9babada0"
PROJ = "/work/11280/zimuq1/vista/LLMmap-project/LLMmap"
WORKDIR = f"{PROJ}/results/D030/zp_prepare"
OUT = f"{PROJ}/confs/baselines/zp_prompts.json"
EMB = f"{PROJ}/confs/baselines/zp_query_emb.npy"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    head = subprocess.check_output(["git", "-C", ZP_REPO, "rev-parse", "HEAD"]).decode().strip()
    assert head == ZP_COMMIT, head
    assert not os.path.exists(f"{WORKDIR}/cache"), "already frozen -- never re-run (Call 9)"
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)                      # official relative cache paths land here
    sys.path.insert(0, ZP_REPO)
    import numpy as np
    import torch
    from utils.utils import load_config, setup_environment
    from fingerprint.fingerprint_factory import create_fingerprint_method

    cfg_path = f"{ZP_REPO}/config/zeroprint.yaml"
    config = load_config(cfg_path)
    args = argparse.Namespace(seed=config.get("seed", 42))
    setup_environment(args)
    zp = create_fingerprint_method(config, accelerator=None)
    zp.prepare()
    ed = zp.embedding_data

    originals = list(ed["original_queries"])
    allq = list(ed["all_queries"])
    pdata = ed["perturbed_queries_data"]
    assert len(originals) == 2 and len(pdata) == 8 and len(allq) == 10
    for q in originals:
        assert q.startswith("Complete the following code: ") and len(q.split()) >= 15, q
    prompts = []
    for i, q in enumerate(allq):
        if i < len(originals):
            prompts.append(dict(prompt_id=i, base=i, variant=0, text=q))
        else:
            d = pdata[i - len(originals)]
            assert d["perturbed_query"] == q
            b = originals.index(d["original_query"])
            o, p = d["original_query"].split(), q.split()
            prompts.append(dict(prompt_id=i, base=b, variant=int(d["perturbed_version"]) + 1, text=q,
                                substitutions=[[a, c] for a, c in zip(o, p) if a != c] if len(o) == len(p)
                                else "token count differs (word_tokenize re-join); see text"))
    for b in range(2):
        assert sorted(p["variant"] for p in prompts if p["base"] == b) == [0, 1, 2, 3, 4]
    E = ed["embeddings"].detach().cpu().float().numpy()
    assert E.shape == (10, 768)
    np.save(EMB, E)
    files = {os.path.relpath(os.path.join(r, f), WORKDIR): sha(os.path.join(r, f))
             for r, _, fs in os.walk(f"{WORKDIR}/cache") for f in fs}
    rec = dict(schema="d030-zp-prompts-v1", zp_repo=ZP_REPO, zp_commit=ZP_COMMIT,
               config_file=cfg_path, config_sha256=sha(cfg_path), config=config,
               seed=args.seed, pythonhashseed=os.environ.get("PYTHONHASHSEED"),
               n_base=2, n_variants=5, n_prompts=10, prompts=prompts,
               all_queries_order=allq, query_embedding_file=os.path.relpath(EMB, PROJ),
               query_embedding_sha256=sha(EMB), official_cache_files=files,
               embedding_model=config["embedding_model_name"],
               torch=torch.__version__)
    rec["sha256_prompts"] = hashlib.sha256(json.dumps(allq, ensure_ascii=False).encode()).hexdigest()
    json.dump(rec, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"[zp-prompts] frozen; sha {rec['sha256_prompts'][:12]}")
    for p in prompts:
        print(p["prompt_id"], p["base"], p["variant"], repr(p["text"]), p.get("substitutions", ""))


if __name__ == "__main__":
    main()
