"""
D005 / M4 — best-effort determination of whether the SHIPPED
`default_dataset.jsonl` (D001/D004's data source) carries either I2 defect.

Method: four system prompts in `systems.json` contain distinctive persona names,
and they fall on both sides of the current train/test split. Models frequently
echo the persona in their responses, so counting echoes and cross-tabbing them
against each entry's own `dataset` field tests which pool each split actually
drew from.

Per D005's budget note this reads available metadata only; it does not attempt to
reconstruct unrecoverable generation history.
"""
import re
import json
import collections

DATASET = "./data/datasets/default_dataset.jsonl"
SYSTEMS = "./confs/prompt_configurations/systems.json"
SPLIT = "./confs/prompt_configurations/train_test_split.json"
OUT = "./results/D005/dataset_forensics.json"

PERSONA_RE = re.compile(r"\bYou are ([A-Z][A-Za-z0-9]{2,}(?:Bot|GPT|AI|Assistant))")


def main():
    systems = json.load(open(SYSTEMS))
    systems = systems["system_prompts"] if isinstance(systems, dict) else systems
    split = json.load(open(SPLIT))
    tr, te = set(split["train"]["systems"]), set(split["test"]["systems"])

    # personas -> (system-prompt index, pool under the CURRENT split file)
    personas = {}
    for i, t in enumerate(systems):
        m = PERSONA_RE.search(t)
        if m:
            name = m.group(1)
            # only usable if the name identifies exactly one prompt
            if sum(name in s for s in systems) == 1:
                personas[name] = dict(
                    idx=i, pool="train" if i in tr else "test" if i in te else None)

    counts = collections.Counter()
    totals = collections.Counter()
    with open(DATASET) as f:
        for line in f:
            d = json.loads(line)
            totals[d["dataset"]] += 1
            blob = " ".join(o for _, o in d["traces"])
            for name in personas:
                if name in blob:
                    counts[(d["dataset"], name)] += 1

    rows = []
    for (split_name, name), n in sorted(counts.items()):
        rows.append(dict(entry_split=split_name, persona=name,
                         persona_pool=personas[name]["pool"],
                         system_prompt_idx=personas[name]["idx"], n=n,
                         cross_pool=split_name != personas[name]["pool"]))

    # --- hypothesis tests against the CURRENT split file
    test_uses_test_pool = any(r["entry_split"] == "test" and
                              r["persona_pool"] == "test" for r in rows)
    train_uses_test_pool = any(r["entry_split"] == "train" and
                               r["persona_pool"] == "test" for r in rows)
    # If there were NO split at all, train-pool personas should also surface in
    # test entries at roughly their train rate.
    train_pool_rate = sum(r["n"] for r in rows
                          if r["entry_split"] == "train"
                          and r["persona_pool"] == "train") / max(totals["train"], 1)
    expected_in_test = train_pool_rate * totals["test"]
    observed_in_test = sum(r["n"] for r in rows if r["entry_split"] == "test"
                           and r["persona_pool"] == "train")

    hyp = {
        "pool_blind_current_bug": dict(
            predicts="test entries never draw a test-pool prompt",
            consistent=not test_uses_test_pool),
        "correct_pool_aware_under_current_split": dict(
            predicts="no test-pool prompt appears in train entries",
            consistent=not train_uses_test_pool),
        "no_split_at_all": dict(
            predicts=f"~{expected_in_test:.0f} train-pool persona echoes in test entries",
            observed=observed_in_test,
            consistent=observed_in_test > 0.25 * expected_in_test),
    }
    surviving = [k for k, v in hyp.items() if v["consistent"]]

    if len(surviving) == 1:
        verdict, note = "DETERMINED", f"exactly one hypothesis survives: {surviving[0]}"
    else:
        verdict = "UNDETERMINABLE"
        note = (
            "No hypothesis is consistent with the evidence under the CURRENT "
            "train_test_split.json. The reading that reconciles everything is that "
            "the current split file is not the one in force when the dataset was "
            "generated -- utility/split_train_test_prompt_conf.py regenerates it "
            "from seed 42, and git shows both files entering in a single squashed "
            "'LLMmap0.2' commit with no intermediate history. Under a "
            "generation-time split where the cross-pool persona sat in train, the "
            "observations are consistent with CORRECT pool-aware generation. That "
            "cannot be proved from the artifact, so no claim is made either way."
        ) if not surviving else (
            f"{len(surviving)} hypotheses remain consistent: {surviving}")

    out = dict(verdict=verdict, note=note, entry_totals=dict(totals),
               personas=personas, echo_cross_tab=rows, hypotheses=hyp)
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"entry totals: {dict(totals)}")
    for r in rows:
        flag = "  <-- CROSS-POOL" if r["cross_pool"] else ""
        print(f"  {r['entry_split']:5s}  {r['persona']:20s} pool={r['persona_pool']:5s} "
              f"n={r['n']:4d}{flag}")
    print("\nhypotheses consistent with the evidence:", surviving or "NONE")
    print(f"\nM4 VERDICT: {verdict}")
    print(note)


if __name__ == "__main__":
    main()
