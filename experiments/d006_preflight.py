"""D006 preflight: which of A1's models need trust_remote_code, lack a chat
template, or fail to load a tokenizer at all. Tokenizer-only, so it is cheap.
Run this BEFORE launching shards -- discovering these one dead job at a time
wastes a slot per model."""
import csv, os, json, sys, glob
from transformers import AutoTokenizer

api = os.environ.get("HUGGINGFACE_API_KEY")
sc = os.environ.get("SCRATCH", "/scratch/11280/zimuq1")
rows = [r["model"] for r in csv.DictReader(open("results/D001/model_metadata.csv"))
        if r["proprietary"].strip().lower() not in ("true", "1", "yes")
        and float(r["params_b"]) <= 14]
out = {}
for m in rows:
    cache = f"{sc}/hf-cache/hub/models--{m.replace('/','--')}"
    staged = bool(glob.glob(cache + "/**/*.safetensors", recursive=True) or
                  glob.glob(cache + "/**/*.bin", recursive=True))
    rec = dict(staged=staged, needs_trust_remote_code=False,
               has_chat_template=None, tokenizer_error=None)
    if staged:
        for trc in (False, True):
            try:
                t = AutoTokenizer.from_pretrained(m, padding_side="left", token=api,
                                                  legacy=False, trust_remote_code=trc)
                rec["needs_trust_remote_code"] = trc
                rec["has_chat_template"] = getattr(t, "chat_template", None) is not None
                rec["tokenizer_error"] = None
                break
            except Exception as e:
                rec["tokenizer_error"] = f"{type(e).__name__}: {str(e)[:110]}"
    out[m] = rec

print(f"{'model':52s} staged trc chat_tpl  error")
for m, r in out.items():
    print(f"{m:52s} {str(r['staged'])[:5]:6s} {str(r['needs_trust_remote_code'])[:5]:3s} "
          f"{str(r['has_chat_template']):8s}  {r['tokenizer_error'] or ''}")
json.dump(out, open("results/D006/preflight.json", "w"), indent=1)
print("\nSUMMARY")
print("  need trust_remote_code:", [m for m,r in out.items() if r["needs_trust_remote_code"]])
print("  NO chat template     :", [m for m,r in out.items() if r["has_chat_template"] is False])
print("  tokenizer fails      :", [m for m,r in out.items() if r["tokenizer_error"]])
print("  not staged           :", [m for m,r in out.items() if not r["staged"]])
