"""D030 / pre-S1 — integrity check of every pinned snapshot (Review P1 A2 + stop-A note).

Run on a `gg` node with the v2 env python. For each of the 85 corpus-v2 models at its manifest
revision, plus mpnet (ZeroPrint's embedder) and e5 (I5): list the repo's files at that revision
over the HF API, keep the files the loaders use (D027's ignore list; `*.bin` kept for repos that
ship no safetensors), and check in `$HF_HOME/hub` that each exists with the API's size and, for
LFS files, the API's sha256 (git-blob sha1 for small files). Anything missing or mismatched is
re-downloaded at the same revision and re-verified; every re-stage is logged.

Output: results/D030/s1_prestage.json
"""
import os
import sys
import json
import time
import fnmatch
import hashlib
from multiprocessing import Pool

from huggingface_hub import HfApi, hf_hub_download

HUB = os.path.join(os.environ["HF_HOME"], "hub")
TOKEN = os.environ["HUGGINGFACE_API_KEY"]
IGNORE = ["consolidated*", "original/*", "*.pth", "*.pt", "*.bin", "*.gguf",
          "*.onnx", "onnx/*", "*.msgpack", "*.h5"]
EXTRA = {"sentence-transformers/all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
         "intfloat/multilingual-e5-large-instruct": "274baa43b0e13e37fafa6428dbc7938e62e5c439"}
OUT = "./results/D030/s1_prestage.json"


def wanted(siblings):
    names = [s.rfilename for s in siblings]
    has_st = any(n.endswith(".safetensors") for n in names)
    keep = []
    for s in siblings:
        ign = [p for p in IGNORE if not (p == "*.bin" and not has_st)]
        if any(fnmatch.fnmatch(s.rfilename, p) for p in ign):
            continue
        keep.append(s)
    return keep


def local_path(repo, rev, f):
    return os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots", rev, f)


def verify(args):
    """args = (repo, rev, file, size, kind, expected); kind = sha256 (LFS) | gitsha1 (git blob)."""
    repo, rev, f, size, kind, expected = args
    p = local_path(repo, rev, f)
    if not os.path.exists(p):
        return (repo, f, "missing")
    n = os.path.getsize(p)
    if size is not None and n != size:
        return (repo, f, "size")
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % n)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return (repo, f, "ok" if h.hexdigest() == expected else "sha")


def main():
    api = HfApi(token=TOKEN)
    man = json.load(open("./confs/baselines/baseline_ext_v1.json"))["models"]
    repos = {m: v["revision"] for m, v in man.items()}
    repos.update(EXTRA)
    tasks, per = [], {}
    for repo, rev in repos.items():
        info = api.model_info(repo, revision=rev, files_metadata=True)
        assert info.sha == rev, (repo, info.sha, rev)
        files = wanted(info.siblings)
        per[repo] = dict(revision=rev, n_files=len(files),
                         gb=round(sum((s.size or 0) for s in files) / 1e9, 2))
        for s in files:
            if s.lfs:
                tasks.append((repo, rev, s.rfilename, s.size, "sha256", s.lfs.sha256))
            else:
                tasks.append((repo, rev, s.rfilename, s.size, "gitsha1", s.blob_id))
    print(f"{len(repos)} repos, {len(tasks)} files, {sum(v['gb'] for v in per.values()):.0f} GB", flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", "32"))) as pool:
        res = pool.map(verify, tasks, chunksize=1)
    bad = [(r, f, s) for r, f, s in res if s != "ok"]
    restaged = []
    for repo, f, why in bad:
        rev = repos[repo]
        hf_hub_download(repo, f, revision=rev, token=TOKEN, force_download=(why != "missing"))
        t = next(x for x in tasks if x[0] == repo and x[2] == f)
        again = verify(t)[2]
        restaged.append(dict(repo=repo, file=f, reason=why, after=again))
        print("restaged", repo, f, why, "->", again, flush=True)
    for repo in per:
        per[repo]["bad_before"] = sum(1 for r, _, _ in bad if r == repo)
        per[repo]["status"] = "OK" if all(x["after"] == "ok" for x in restaged if x["repo"] == repo) else "FAILED"
    out = dict(checked=time.strftime("%Y-%m-%dT%H:%M:%S"), n_repos=len(repos), n_files=len(tasks),
               wall_s=round(time.time() - t0, 1), n_bad_before=len(bad), restaged=restaged,
               all_ok=all(v["status"] == "OK" for v in per.values()), repos=per,
               note="integrity check required by Review D030/P1 A2 before S1; not a purge-clock touch")
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "repos"}, indent=1))
    sys.exit(0 if out["all_ok"] else 1)


if __name__ == "__main__":
    main()
