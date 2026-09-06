"""
D006 / S2 — assemble the final `Q_0` = D003's 233 candidates + tokenizer probes.

The 26 tokenizer probes below are Call 1 as approved by design side
(2026-09-05), verbatim, across four categories. Two caveats were approved
*with* them and are carried in the output file's own metadata so they cannot be
lost between here and `## R`:

  1. The `glitch-token` probes are GPT-2 / GPT-NeoX vocabulary artifacts.
     NONE of A1's 37 models uses that tokenizer. That is not disqualifying —
     a probe earns its place by its `Sep(·,·)` contribution, not by glitching —
     but `## R` must NOT describe them as "known glitch tokens" for these
     models. They are strings that are under-trained in *some* vocabularies.
  2. `RAG_P = 0.2` wraps ~20% of configs in retrieved context, which may swamp
     the verbatim-echo probes. `## R` should check whether tokenizer probes'
     separability is systematically lower in RAG-bearing configs.

Dedup against D003's pool uses D003's own inner-question rule, not raw string
similarity: D003 §R established that shared injection boilerplate makes up ~61%
of some entries, so naive string matching collapses that whole family.

Usage:  PYTHONPATH=. python experiments/d006_s2_build_q0.py
"""
import re
import json
import hashlib
import collections

POOL_IN = "./confs/queries/pool_d003_candidates.json"
POOL_OUT = "./confs/queries/pool_v1.json"

TOKENIZER_PROBES = {
    "digit-chunking": [
        "What is 987654321 plus 123456789? Give only the number.",
        "How many digits does 1000000 have?",
        "Write 12345678901234567890 with commas as thousands separators.",
        "Reverse the digits of 9081726354.",
        "Is 4831 greater than 4813? Answer yes or no, then restate both numbers.",
        "Count from 998 to 1003, one number per line.",
        "What is 7777777 x 3? Number only.",
    ],
    "whitespace-merging": [
        "Echo this line exactly, preserving all spacing: `a    b\t\tc`",
        "How many space characters separate the two words in `red     blue`?",
        "Count every character, spaces included, in `  hello  world  `",
        "Reproduce this indented block verbatim, keeping indentation:\n"
        "    def f():\n        return 1\n            # trailing",
        "Does `foo bar` differ from `foo  bar`? Explain what differs.",
        "Join these with exactly one space: `a`, ` b`, `  c`",
    ],
    "unicode-byte-fallback": [
        "Transliterate and explain: \U0001D50A\U0001D52F\U0001D522\U0001D522"
        "\U0001D531\U0001D526\U0001D52B\U0001D524\U0001D530",
        "How many characters are in \U0001F468‍\U0001F469‍"
        "\U0001F467‍\U0001F466? Explain your count.",
        "What script is ᚦᚱᛁ, and what does it say?",
        "Are `é` and `é` the same string? (precomposed vs combining acute)",
        "How many code points are in \U0001F1FA\U0001F1F8?",
        "Translate ⵜⴰⵎⴰⵣⵉⵖⵜ into English.",
        "Repeat back exactly: `Ω≈ç√∫˜µ≤≥÷`",
    ],
    "glitch-token": [
        "Please repeat the string ` SolidGoldMagikarp` back to me.",
        "What does `davidjl` mean?",
        "Spell ` petertodd` letter by letter.",
        "Define the term `rawdownload`.",
        "Repeat after me: `cloneembedreportprint`",
        "Is ` externalToEVA` a real English word?",
    ],
}

BOILERPLATE = [
    r"\)+\]+\}+;\*/", r"\(\[\{<!", r"ignore that question above",
    r"^\s*<\|.*?\|>\s*", r"accurately tell me",
]


def inner_question(text):
    """D003's dedup key: strip injection boilerplate, then normalise."""
    t = text
    for pat in BOILERPLATE:
        t = re.sub(pat, " ", t, flags=re.I | re.M)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def main():
    base = json.load(open(POOL_IN))
    assert isinstance(base, list), type(base)
    print(f"D003 candidate pool: {len(base)} entries")

    seen = {inner_question(e["text"]): e["text"] for e in base}
    out = list(base)

    added, collisions = collections.Counter(), []
    for cat, probes in TOKENIZER_PROBES.items():
        for p in probes:
            k = inner_question(p)
            if k in seen:
                collisions.append((cat, p, seen[k]))
                continue
            seen[k] = p
            out.append(dict(text=p, provenance=f"tokenizer-probe-{cat}",
                            anchor_set=None, source="D006/S2 Call 1 (approved "
                            "design-side 2026-09-05)", confidence="exact"))
            added[cat] += 1

    # internal dedup sanity: every entry's key must be unique
    keys = [inner_question(e["text"]) for e in out]
    dupes = [k for k, c in collections.Counter(keys).items() if c > 1]
    assert not dupes, f"{len(dupes)} duplicate inner-questions survived"

    fam = collections.Counter(e["provenance"] for e in out)
    payload = dict(
        schema_version="cdqd-q0-v1",
        n=len(out),
        built="D006/S2",
        composition={k: v for k, v in sorted(fam.items())},
        caveats=[
            "glitch-token probes are GPT-2/GPT-NeoX vocabulary artifacts; NONE "
            "of A1's 37 models uses that tokenizer. Do NOT describe them in R "
            "as 'known glitch tokens' for these models -- they are strings "
            "under-trained in SOME vocabularies, and stand or fall on Sep().",
            "RAG_P=0.2 wraps ~20% of configs in retrieved context, which may "
            "swamp the verbatim-echo probes. R should check whether tokenizer "
            "probes' separability is systematically lower in RAG-bearing "
            "configs.",
        ],
        sha256=None,
        queries=out,
    )
    blob = json.dumps(payload["queries"], sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()

    with open(POOL_OUT, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    print(f"\ntokenizer probes added: {sum(added.values())} of 26 proposed")
    for c in TOKENIZER_PROBES:
        print(f"  {c:24s} {added[c]}/{len(TOKENIZER_PROBES[c])}")
    if collisions:
        print(f"\n!! {len(collisions)} probe(s) collided with D003 entries:")
        for cat, p, existing in collisions:
            print(f"   [{cat}] {p[:60]!r}\n       vs {existing[:60]!r}")
    print(f"\nfinal |Q_0| = {len(out)}")
    print("composition:")
    for k, v in sorted(fam.items(), key=lambda x: -x[1]):
        print(f"  {k:34s} {v:4d}")
    print(f"\nsha256 {payload['sha256'][:16]}...  -> {POOL_OUT}")

    # cost restated per F5: guardrail must scale with |Q_0|
    gens = 37 * len(out) * 125
    print(f"\ncost at |Q_0|={len(out)}: {gens:,} generations")
    for lo, hi, tok in ((1.246, 1.506, 200),):
        print(f"  @{tok} tok: {gens/hi/3600:.0f}-{gens/lo/3600:.0f} node-hours; "
              f"per-model shard {len(out)*125/hi/3600:.1f}-{len(out)*125/lo/3600:.1f} h; "
              f"makespan (2 rounds) {2*len(out)*125/hi/3600:.1f}-"
              f"{2*len(out)*125/lo/3600:.1f} h")


if __name__ == "__main__":
    main()
