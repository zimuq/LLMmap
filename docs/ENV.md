# ENV.md — TACC Vista operational facts

**Owner: TACC-side.** Machine facts and operational discipline — no research
decisions, no findings. If a fact here is wrong, the machine is the authority;
re-run the verification command and correct it.

**Why this file exists.** Every fact below was previously recorded only inside a
*closed* D's plan file, or in conversation that no file captured. `D006.md` cites
a "TACC 2026-09-04 report" three times as the source of the `gh` wall cap and job
limits; that report is a chat message, not a document, and does not exist in the
repo. A session whose context has been compacted cannot recover any of it. This
file is the recovery point: **read this before submitting any job.**

Last verified: **2026-09-05** (commands given so it can be re-verified, not trusted).

---

## Partitions and QoS

`scontrol show partition <p>` · `sacctmgr -nP show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU`

| partition | nodes | QoS | **max wall** | **max jobs running / submitted per user** | max nodes per user |
|---|---|---|---|---|---|
| `gh` | 576 | `qgh` | **2 days** | **20 / 40** | 96 |
| `gh-dev` | 20 | `qdevelopment` | **2 h** | **1 / 3** | 8 |
| `gg` | 251 | `qgg` | 2 days | 20 / 40 | 96 |
| `gb` | 18 | `qgb` | 12 h | 2 / 3 | 18 |

`sinfo` reports `TIMELIMIT=infinite` for every partition. **That is misleading** —
the real cap is the QoS `MaxWall` above, and a job asking for more is rejected at
submit time. Use the QoS column.

Consequences that have already shaped a plan:

- **A single job cannot exceed 2 days on `gh`.** Any workload longer than that
  must be sharded — this is a hard constraint, not a tuning preference.
- **`MaxSubmitPU = 40` on `gh`**, so up to 40 jobs may sit in the queue while 20
  run. Sharding a job 37 ways therefore needs no hand-scheduled "waves": submit
  all 37 and Slurm backfills as slots free.
- `gh-dev` allows **one job at a time** and 2 h. Fine for smoke tests and
  short analyses (D001–D005 all used it); unusable for corpus generation.

**Vista rejects `--gres` and `--gpus-per-task`.** The scheduler errors out on
these. Absence of GRES does **not** mean absence of a GPU — every `gh`/`gh-dev`
node has one and `torch.cuda.is_available()` is `True` there. (This tripped
D002/P1 §F5 and was corrected in D002's `## R`.) Request nodes with `-N`/`-n`
only.

## Hardware

- `gh`/`gh-dev` node = **NVIDIA GH200**, 72-core aarch64 Grace + Hopper GPU.
- **Memory: `torch` sees 95.0 GB** (`results/D002/gpu_verification.json`, measured).
  Vendor material calls the same part "GH200 120GB" and some notes here say
  "H200 96GB". **95 GB is the number to plan against** — it is the measured,
  allocatable figure.
- `gg` nodes are CPU-only.
- Charging: **1 SU per node-hour**, 15-minute minimum per job.

## Filesystems

| path | quota | purge | use for |
|---|---|---|---|
| `$HOME` (`/home1/…`) | 23.3 GB | none | nothing large |
| `$WORK` (`/work/11280/zimuq1/vista`) | **1 TB**, shared across all TACC systems | **none** | code, corpora, embeddings, results — anything expensive to regenerate. **Not backed up.** |
| `$SCRATCH` (`/scratch/11280/zimuq1`) | **no quota** | **purged after 10 days without compute-node access** | model weights, `HF_HOME` — anything re-downloadable |

The purge clock is reset by access **from a compute node**, not from a login node.
A long build that touches weights only at the start can still lose them mid-run.

Check current usage: `/usr/local/etc/taccinfo` (also prints the SU balance — read
it live rather than trusting a number written in a doc; it was 6982 when D002 ran
and 6869 on 2026-09-05).

**The SU balance is NOT a measure of your own usage.** The allocation
(`TG-NAIRR250513`) is shared with other group members, so the balance moves for
reasons that have nothing to do with your jobs. During D006 the balance fell 123
SU while this project's own jobs accounted for 44.6 — the remainder was other
people. **To measure your own cost, use `sacct`, not the balance:**

```bash
sacct -S <start-date> -X -n -o JobID,JobName%28,State,Elapsed,ElapsedRaw
```

and apply the 15-minute-per-job minimum. D006 briefly and wrongly suspected the
documented 1 SU/node-hour rate was inflated by ~2.7x on the strength of the
balance delta; the rate is correct, the balance was simply not measuring the
right thing.

## Conda environments

| env | torch | use |
|---|---|---|
| `llmmap` | `2.7.1+cpu` | CPU-only analysis |
| **`llmmap-gpu`** | **`2.7.1+cu128`** | **everything on a GPU node** |
| `envs/llmmap-internlm` | inherits | **only** `internlm2_5-7b-chat` (see below) |

**`llmmap-gpu` is the one you want.** Activating `llmmap` on a GPU node silently
runs on CPU — no error, just ~50× slower.

Installing CUDA torch here has two traps, both hit once already:

- Use the **cu128** index, not cu126 — there is no aarch64 `2.7.1` wheel on cu126,
  so pip resolves to `2.14.0` and `requirements.txt` then downgrades it to the CPU
  wheel.
- Pin the **local version explicitly**: `pip install torch==2.7.1+cu128`.
  `torch==2.7.1` is a silent no-op when `2.7.1+cpu` is installed, because PEP 440
  treats `2.7.1+cpu` as satisfying `==2.7.1`.

### The `llmmap-internlm` side environment

`internlm/internlm2_5-7b-chat` cannot load under `llmmap-gpu`, and the reason is
worth recording because the first error message pointed at the wrong subsystem.
`AutoTokenizer` reports a failed SentencePiece->fast conversion, which reads like
a transformers problem. It is not: **`sentencepiece` 0.2.x rejects internlm2's
vocabulary** (its pieces contain null bytes; 0.1.99 accepts them). The local
`tokenizer.model` was verified byte-identical to remote, so it is not corruption.

**Do not fix this by downgrading `llmmap-gpu`.** Verified: under sp 0.1.99,
`EuroLLM-1.7B` and `Mistral-7B-v0.3` both fail with protobuf descriptor errors.

Recipe (a venv over `llmmap-gpu`, so transformers/torch stay identical and the
deviation is exactly two packages):

```bash
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
python -m venv --system-site-packages /work/11280/zimuq1/vista/envs/llmmap-internlm
/work/11280/zimuq1/vista/envs/llmmap-internlm/bin/pip install \
    "sentencepiece==0.1.99" einops
```

`einops` is required by internlm's own remote modeling file. The model also needs
`trust_remote_code=True` **and** `use_fast=False`. Run it by invoking that venv's
python directly (see `experiments/submit_d006_internlm.slurm`); do not activate it
over another env.

## Never compute on a login node

Login nodes cap `ulimit -u` at 100 processes. Tokenizer/rayon thread pools panic,
and load average has been driven to 48 by a single 20-sample job. Use `gh-dev` for
anything interactive. Pre-staging model downloads from a login node is fine and
is the recommended pattern — compute-node egress to the HF Hub is untested.

## Working sbatch skeleton

Copy this; every literal path below is required and appears nowhere else in `docs/`.

```bash
#!/bin/bash
#SBATCH -J <jobname>
#SBATCH -p gh              # gh-dev for <=2h single-job tests
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00        # must be <= QoS MaxWall (gh: 2 days)
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/<D>/slurm-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/<D>/slurm-%j.err

source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache
export HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)

echo "=== node: $(hostname) start: $(date) ==="
python experiments/<script>.py
echo "=== end: $(date) ==="
```

`HUGGINGFACE_API_KEY` is required by `llm.py:23` **even for ungated repos** — the
constructor raises without it.

## Measured throughput (D002 §R2, batch-8, GH200)

| tokens | gen/s | note |
|---|---|---|
| 100 | 2.255 – 2.478 | measured on Qwen2.5-3B and 7B |
| 200 | 1.246 – 1.506 | |

Batching gives **3.7–6.9×** over the batch-1 path. D002's own caveat, which
matters when extrapolating: **throughput does not track parameter count** here
(the 7B was *faster* than the 3B unbatched — 28 vs 36 layers; at small batch,
decoding is launch-latency-bound, not compute-bound). Treating the rate as flat
across 0.5B–14B is a reasonable extrapolation but **is** an extrapolation — only
3B and 7B were measured.

**Estimating wall-clock:** with per-model shards, the makespan is
`ceil(n_models / 20) × per-shard-hours`, **not** `total-node-hours / 20`. The
latter is a utilisation figure and understates makespan by ~2× at n=37. This error
is on record (`D006.md`'s "4–6 hours"; corrected in `plans/D006-P1.md` §F4).

---

<!-- Append machine facts as they are established. Keep decisions and findings
     out of this file — they belong in DECISIONS.md / FINDINGS.md. -->

---

## Operational discipline (Phase 1, learned the expensive way)

Five mistakes recurred across D006/D007/D008. Each cost real time, none was caught by
"does the code do what I intended", and all four are cheap to avoid. They are
recorded here rather than inside a closed D because they are not about those D's
questions — they are about how to run work on this machine.

**1. Set thresholds at what a DEFECT looks like, not at "perfect."** Five false
alarms came from this, each costing a diagnostic detour:

| check | set at | should have been |
|---|---|---|
| batched vs unbatched generation | character-exact match | distributional — bf16 makes exactness impossible |
| batch-size effect on length | 32 prompts, chars | paired, tokens, both size extremes |
| shard validity | zero empty responses | anomalous *rate* (empties are model behaviour) |
| point-cloud health | ≥80% unique rows | collapsed spread (repetition is signal) |
| D007's F2 | assumed bias would flip T1.6 | it was real but 10× too small; the *audit gate* mattered |

An over-strict criterion does not fail safe. It manufactures alarms that look
exactly like real defects.

**2. Guards must match how the thing actually fails.** `AutoTokenizer` returns a
**`bool`** instead of raising for some remote-code models. A `try/except` around
it catches nothing, and the loop dies 16 models later on `'bool' object is not
callable`. I had documented that exact behaviour myself and then written the
wrong guard. Check the returned object is usable; don't assume failure arrives as
an exception. Likewise `set -euo pipefail` turned a zero-match `grep` — a normal
outcome — into a silent total failure of the submitter.

**3. Never mutate shared state while jobs are queued or running.** Rewriting
`general.json` killed a PENDING job 90 s in. Installing packages into
`llmmap-gpu` mid-run left 24 shards with unrecoverable environment provenance.
Use a venv with `--system-site-packages` for one-off needs (see above).

**5. Compare floats in the dtype you intend, not the one the array happens to
carry.** D008's `CVaR(x, 1.0) == mean(x)` correctness assertion failed on the
first run at 2.8e-8 — the tensor is stored float32, `cvar` sorts before averaging,
and `np.mean` of the same float32 values sums in a different order. Nothing about
the reduction being tested was wrong. Cast both sides to float64 at the boundary
where a numerical identity is asserted. (Same family as the bf16 non-associativity
that broke the batched-generation check; see item 1.)

**4. Prefer a job's own report over inference from job names.** Slurm job names
are truncated to 20 chars, so `granite-3.0`/`granite-3.1`,
`Phi-3-medium-128k`/`-4k` and `Mistral v0.1/v0.2/v0.3` collide. De-duplicating on
them cancelled four healthy jobs. Read `--model` back out of the batch script
(`scontrol write batch_script <id> -`).

**Two things that paid for themselves repeatedly:** a *preflight* that checks
every model's tokenizer/template/quirks before launching N jobs, and a *smoke
test* on the smallest and largest model before committing a fleet. Both were
written only after failures they would have prevented.
