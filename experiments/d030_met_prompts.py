"""D030 / S0.3 — freeze MET's 25 prompts (P1 Call 1, approved: MET's released draw).

Run with envs/d030-met/bin/python (datasets + the pinned MET repo). The prompt text is
produced by MET's own builder, `experiments.prompts._get_wikipedia_prompts`, which streams
the first 100 de-duplicated rows of `Cohere/wikipedia-2023-11-embed-multilingual-v3` per
language (`StreamingDataset(ds, 100)`). The chat-format columns it also computes need a
model object; a stub returns "" for them -- only `plain` is kept, our wrapper renders the
template per model at generation time (P1 S0.5(c) checks parity with MET's renderer).

Draw: experiments/constants/wikipedia_prompt_indices_test/25/0.json, file order
(en, de, fr, ru, es; ascending index) -> prompt ids 0..24.
Also asserts (P1 Call 2) that the parquet data files are those of the 2024-01 upload.

Output: confs/baselines/met_prompts.json
"""
import os
import sys
import json
import hashlib

MET_REPO = "/work/11280/zimuq1/vista/ext/model-equality-testing"
MET_COMMIT = "fd2ee24d75c9fef87debff8caefa0c04d4a5d374"
DRAW = "experiments/constants/wikipedia_prompt_indices_test/25/0.json"
DS = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
OUT = "./confs/baselines/met_prompts.json"


class _StubModel:
    def format_as_chat(self, plain, tokenize=False, **kw):
        return [] if tokenize else ""


def parquet_oids(langs):
    """P1 Call 2: the parquet files MET streamed in Jul-Aug 2024 must be the ones at today's
    main. (1) By path: main vs the last commit before MET's sampling window (2024-03-19,
    README-only after the 2024-01-26 'add'). (2) By content: main vs the 2024-01-11 upload,
    keyed by language/basename -- the 2024-01-25 'renaming' commit moved `20231101.<lang>/`
    to `<lang>/` without changing any LFS sha (found in S0; recorded)."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HUGGINGFACE_API_KEY"))
    commits = [dict(id=c.commit_id, date=c.created_at.isoformat(), title=c.title)
               for c in api.list_repo_commits(DS, repo_type="dataset")]
    upload = [c for c in commits if c["title"] == "add parquet files"]
    before_met = [c for c in commits if c["date"] < "2024-07-01"]
    assert len(upload) == 1 and before_met

    def oids(rev):
        info = api.dataset_info(DS, revision=rev, files_metadata=True)
        out = {}
        for s in info.siblings:
            if not s.rfilename.endswith(".parquet"):
                continue
            d, f = s.rfilename.split("/")
            lang = d.split(".")[-1]
            if lang in langs:
                out[(s.rfilename, f"{lang}/{f}")] = s.lfs.sha256 if s.lfs else None
        return info.sha, out

    main_sha, now = oids("main")
    _, pre = oids(before_met[0]["id"])
    _, up = oids(upload[0]["id"])
    assert now and {k[0]: v for k, v in now.items()} == {k[0]: v for k, v in pre.items()}, \
        "parquet files differ from the pre-MET-sampling revision -> Call"
    assert {k[1]: v for k, v in now.items()} == {k[1]: v for k, v in up.items()}, \
        "parquet content differs from the 2024-01 upload -> Call"
    return dict(sha=main_sha, upload_commit=upload[0]["id"], pre_met_commit=before_met[0]["id"],
                files={k[0]: v for k, v in sorted(now.items())}, commits=commits,
                titles_after_upload=sorted({c["title"] for c in commits if c["date"] > upload[0]["date"]}))


def main():
    sys.path.insert(0, MET_REPO)
    import subprocess
    head = subprocess.check_output(["git", "-C", MET_REPO, "rev-parse", "HEAD"]).decode().strip()
    assert head == MET_COMMIT, head
    from experiments import prompts as P
    draw = json.load(open(os.path.join(MET_REPO, DRAW)))
    langs = [k.split("_")[1] for k in draw]
    assert sum(len(v) for v in draw.values()) == 25
    out = []
    for key, idx in draw.items():
        lang = key.split("_")[1]
        ds = P._get_wikipedia_prompts(_StubModel(), lang)
        assert len(ds.samples) == 100, (lang, len(ds.samples))
        for i in sorted(idx):
            row = ds.samples[i]
            plain = row["plain"]
            assert plain.startswith("Continue the paragraph. Do not output anything except the "
                                    "continuation to the paragraph. Start the continuation immediately.\n\"")
            assert plain.endswith('..."')
            out.append(dict(prompt_id=len(out), source=key, row_index=i, met_id=row["id"], plain=plain))
    ver = parquet_oids(langs)
    rec = dict(schema="d030-met-prompts-v1", met_repo=MET_REPO, met_commit=MET_COMMIT, draw_file=DRAW,
               draw=draw, dataset=DS, dataset_revision_at_freeze=ver["sha"],
               parquet_lfs_sha256=ver["files"], dataset_commits=ver["commits"],
               dataset_commit_titles_after_upload=ver["titles_after_upload"],
               dataset_upload_commit=ver["upload_commit"], dataset_pre_met_commit=ver["pre_met_commit"],
               parquet_unchanged_since_upload="by content (renamed 2024-01-25); by path since pre-MET commit",
               n_prompts=len(out), prompts=out)
    rec["sha256_prompts"] = hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False)
                                           .encode()).hexdigest()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"[met-prompts] {len(out)} prompts, sha {rec['sha256_prompts'][:12]}, dataset rev {ver['sha'][:10]}")
    for p in out:
        print(p["prompt_id"], p["source"], p["row_index"], repr(p["plain"][-110:]))


if __name__ == "__main__":
    main()
    # datasets' streaming threads abort the interpreter at shutdown (PyGILState_Release,
    # seen in S0 job 1060194 after the file was written); exit without teardown.
    sys.stdout.flush()
    os._exit(0)
