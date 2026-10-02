"""D026 — shared pieces: the pre-registered probe list, the F3 filler, and the
D006-harness loader for the two Phi-3-medium twins.

The probe texts below are copied verbatim from `docs/D026.md` `## D` ("The
probes — fixed here, before any generation"). They are the pre-registration:
do not edit them here. A problem with one is a Call, not an edit.

The only text this file *constructs* is the F3 assembly (needle + filler +
question); its format and the filler rule are disclosed in
`docs/plans/D026-P1.md` and recorded in `results/D026/s0.json`.
"""
import os
import re
import json
import hashlib

TWINS = ["microsoft/Phi-3-medium-128k-instruct",
         "microsoft/Phi-3-medium-4k-instruct"]
CORPUS_DIR = "./data/corpus_v1"
OUT = "./results/D026"
FIRST_NEW_ID = 259

# ---------------------------------------------------------------- probes (## D)
F1 = [
    "What is the maximum context length you support, in tokens?",
    "How long can my input be before you start losing track of its beginning? Give a number of tokens.",
    "Is your context window 4K tokens or 128K tokens? Answer with just one of the two.",
    "If I send you a 300-page book in one message, can you read all of it? Answer yes or no, then state your context limit.",
    "Describe your context window in one sentence.",
    "Were you trained for long-context tasks? Answer briefly.",
    "What is the full name of your model, including any context-length suffix?",
    "Roughly how many words can you take as input at once?",
]
F2 = [
    "How many parameters do you have? Answer with a single number.",
    "Which company trained you, and what is your exact model version?",
    "Are you a small, medium, or large language model? Explain in one sentence.",
    "What architecture are you based on, and how many layers do you have?",
    "What is your knowledge cutoff date?",
    "List three facts about yourself: your name, your size, and your context length.",
]
# `## D` writes the needles as "The secret word is *marmalade*." in Markdown.
# P1 Call 2 reads the asterisks as Markdown emphasis, not probe text; the
# reading is a parameter here so the Review can flip it without an edit below.
NEEDLE_WORDS = ["marmalade", "lighthouse"]
NEEDLE_LITERAL_ASTERISKS = False
QUESTION = ("What was the secret word mentioned at the very beginning of this "
            "message? Answer with one word.")
F3_LENGTHS = [2000, 6000, 12000]          # 273/274, 275/276, 277/278
SEP = "\n\n"                              # needle SEP filler SEP question

FILLER_SRC = f"{OUT}/filler/pg84.txt"
FILLER_META = dict(
    title="Frankenstein; or, the Modern Prometheus", author="Mary Wollstonecraft Shelley",
    source="Project Gutenberg eBook #84",
    url="https://www.gutenberg.org/cache/epub/84/pg84.txt",
    edition="Release date: October 1, 1993 [eBook #84]; Most recently updated: February 10, 2026",
    retrieved="2026-10-01",
    sha256="7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3",
    rule=("CRLF->LF; keep the text from the first line equal to 'Letter 1' (the "
          "body, after the contents list) up to the '*** END OF' marker; split on "
          "blank lines into paragraphs; join each paragraph's hard-wrapped lines "
          "with one space and strip; join paragraphs with '\\n\\n'. Tokenise with "
          "the twins' shared tokenizer (add_special_tokens=False), keep the first "
          "L token ids, decode."),
)


def needle(word):
    w = f"*{word}*" if NEEDLE_LITERAL_ASTERISKS else word
    return f"The secret word is {w}."


def filler_text():
    raw = open(FILLER_SRC, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == FILLER_META["sha256"], \
        "filler source changed on disk"
    lines = raw.decode("utf-8").replace("\r\n", "\n").split("\n")
    start = lines.index("Letter 1")
    end = next(i for i, l in enumerate(lines) if l.startswith("*** END OF"))
    body = "\n".join(lines[start:end])
    paras = [" ".join(l.strip() for l in p.split("\n")).strip()
             for p in re.split(r"\n\s*\n", body)]
    return "\n\n".join(p for p in paras if p)


def f3_probes(tokenizer):
    """[(text, meta)] for ids 273..278, in the D's order."""
    ids = tokenizer(filler_text(), add_special_tokens=False)["input_ids"]
    out = []
    for L in F3_LENGTHS:
        assert len(ids) >= L
        fill = tokenizer.decode(ids[:L])
        for w in NEEDLE_WORDS:
            text = SEP.join([needle(w), fill, QUESTION])
            out.append((text, dict(
                L=L, needle=w,
                filler_tokens_retokenised=len(tokenizer(fill, add_special_tokens=False)["input_ids"]),
                bare_probe_tokens=len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                filler_secret_mentions=len(re.findall(r"secret", fill, re.I)),
                filler_contains_needle_word=any(n in fill.lower() for n in NEEDLE_WORDS))))
    return out


def all_probes(tokenizer):
    """{id: text} for 259..278 and the F3 meta."""
    f3 = f3_probes(tokenizer)
    texts = F1 + F2 + [t for t, _ in f3]
    assert len(texts) == 20
    return ({FIRST_NEW_ID + i: t for i, t in enumerate(texts)},
            {FIRST_NEW_ID + 14 + i: m for i, (_, m) in enumerate(f3)})


FAMILY = {**{i: "F1" for i in range(259, 267)},
          **{i: "F2" for i in range(267, 273)},
          **{i: "F3" for i in range(273, 279)}}


# ---------------------------------------------------------------- corpus side
def slug(m):
    return m.replace("/", "__")


def shard_status(m):
    return json.load(open(f"{CORPUS_DIR}/{slug(m)}.status.json"))


def corpus_entries(m):
    """{(pool, config_index): entry} from corpus_v1 (traces included)."""
    out = {}
    with open(f"{CORPUS_DIR}/{slug(m)}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            out[(d["dataset"], d["config_index"])] = d
    assert len(out) == 125, (m, len(out))
    return out


def greedy_reference(m, entries):
    """D024's tier-1 SINGLE config under the 'effective' reading (greedy counts
    as temperature 0). Values verified against results/D024/s0_pilot.json."""
    ref = json.load(open("./results/D024/s0_pilot.json"))
    c = ref["single_tier"]["readings"]["effective"][m]["config_index"]
    pc = entries[("build", c)]["prompt_conf"]
    assert pc["sampling_hparams"]["do_sample"] is False
    return c


# ---------------------------------------------------------------- harness
def load_twin(m, revision):
    """Load exactly as `d006_s4_shard.main` does for a model that needs no
    quirk (neither twin is in TRUST_REMOTE_CODE / EXTRA_TOKENIZER_KWARGS /
    CHAT_TEMPLATE_FALLBACK -- asserted), pinned to the shard's recorded
    revision. Returns (llm, provenance)."""
    import torch
    from LLMmap.llm import LLM_huggingface
    import d006_s4_shard as h
    assert m not in h.TRUST_REMOTE_CODE and m not in h.EXTRA_TOKENIZER_KWARGS \
        and m not in h.CHAT_TEMPLATE_FALLBACK
    llm = LLM_huggingface(m, model_load_kargs=dict(torch_dtype=torch.bfloat16,
                                                   device_map="cuda",
                                                   revision=revision))
    got = getattr(llm.model.config, "_commit_hash", None)
    assert got == revision, f"{m}: loaded {got}, corpus used {revision}"
    gc = llm.model.generation_config
    use_cache_overridden = False
    if getattr(gc, "use_cache", True) is False:
        gc.use_cache = True
        use_cache_overridden = True
    import transformers
    cfg = llm.model.config
    prov = dict(hf_revision=got, use_cache_overridden=use_cache_overridden,
                attn_implementation=getattr(cfg, "_attn_implementation", None),
                sliding_window=getattr(cfg, "sliding_window", None),
                max_position_embeddings=cfg.max_position_embeddings,
                original_max_position_embeddings=getattr(cfg, "original_max_position_embeddings", None),
                rope_type=((cfg.rope_scaling or {}).get("rope_type") or (cfg.rope_scaling or {}).get("type")),
                supports_system_role=llm.supports_system_role,
                transformers=transformers.__version__, torch=torch.__version__,
                batch=h.CORPUS_BATCH, token_ceiling=h.TOKEN_CEILING)
    return llm, prov
