"""D027 — prompting/loading wrapper for the screen (P1 Calls 2-4, approved).

Importable in BOTH envs (`llmmap-gpu`, transformers 4.51; `envs/llmmap-gpu-v2`,
transformers 5.18): it imports only LLMmap.llm and transformers.

Differences from `LLMmap.llm.LLM_huggingface` (whose logic is otherwise
reproduced line for line -- left padding, pad = eos, the system-role sentinel,
system-prompt prepending when the template has no system role):
  * every chat-template render gets TEMPLATE_KWARGS:
      enable_thinking=False, thinking=False   (Call 4)
      strftime_now = frozen to 2026-09-06     (Call 3)
    `date_string` is deliberately NOT passed: Llama-3.1-style templates default
    it to the fixed string "26 Jul 2024" (that is what corpus_v1 rendered for
    Meta-Llama-3.1-8B); passing it would change a v1 prompt. Llama-3.2-style
    templates derive date_string from strftime_now, which is frozen. (S0
    correction to P1 Call 3, disclosed.)
  * weights pinned to a revision; `dtype` instead of the deprecated
    `torch_dtype`; tokenizer-only kwargs never reach the model constructor.
  * VLM-class checkpoints (Gemma3/Gemma4/Mistral3/Qwen3.5 `...ForConditional
    Generation`): AutoModelForCausalLM first; if that refuses the config, the
    checkpoint's own `architectures[0]` class, text-only inputs.
  * no trust_remote_code except D006's allowlist (v1 only).
"""
import os
import datetime

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from LLMmap.llm import LLM_huggingface

FROZEN_DATE = datetime.datetime(2026, 9, 6)


def _strftime_now(fmt):
    return FROZEN_DATE.strftime(fmt)


TEMPLATE_KWARGS = dict(enable_thinking=False, thinking=False, strftime_now=_strftime_now)
TEMPLATE_KWARGS_RECORD = dict(enable_thinking=False, thinking=False,
                              strftime_now=f"frozen:{FROZEN_DATE.date().isoformat()}")
# S0 compatibility shims (transformers 5 vs the v1 env), each found by the
# tokenisation-parity / load checks and recorded per model in load_info:
#  * Llama-2-7b-chat: v1's fast tokenizer IGNORED legacy=False and inserted
#    a prefix space after <s> ('▁[' = 518); transformers 5 honours
#    legacy=False ('[' = 29961), changing the first token of every prompt.
#    legacy=True reproduces v1 exactly (verified by d027_tokparity).
#  * LongRoPE configs whose rope_scaling lacks original_max_position_embeddings
#    (present at top level): transformers 5's validator raises KeyError. The
#    shim copies the top-level value (4096) into the rope dict; same number
#    4.51 used.
TOKENIZER_COMPAT_V5 = {"meta-llama/Llama-2-7b-chat-hf": {"legacy": True}}

THINK_MARKERS = ["<think", "</think>", "<|think|>", "<|channel|>", "<|channel>", "<reasoning"]


def render(tokenizer, supports_system, system, user, tkw=TEMPLATE_KWARGS):
    """LLM_huggingface.make_prompt, with template kwargs."""
    messages = []
    if system:
        if supports_system:
            messages.append({"role": "system", "content": system})
        else:
            user = f"{system}\n\n{user}"
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, **tkw)


def has_system_role(tokenizer, tkw=TEMPLATE_KWARGS):
    s = LLM_huggingface._SYS_SENTINEL
    try:
        out = tokenizer.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": "probe"}],
            tokenize=False, add_generation_prompt=True, **tkw)
    except Exception:
        return False
    return s in out


class LLMv2(LLM_huggingface):
    """Same public surface as LLM_huggingface (make_prompt / generate), so
    PromptConf.__call__ and make_dataset_entries_for_new_llm work unchanged."""

    def __init__(self, llm_name, revision, tokenizer_only=False, trust_remote_code=False,
                 tokenizer_kwargs=None, chat_template_fallback=None):
        token = os.environ.get("HUGGINGFACE_API_KEY")
        assert token, "HUGGINGFACE_API_KEY missing (llm.py requires it)"
        self.llm_name = llm_name
        self.revision = revision
        self.is_hf = True
        self.load_info = dict(transformers=transformers.__version__, torch=torch.__version__)
        tk = dict(padding_side="left", token=token, legacy=False, revision=revision)
        if trust_remote_code:
            tk["trust_remote_code"] = True
        tk.update(tokenizer_kwargs or {})
        self.load_info["compat_shims"] = []
        v5 = int(transformers.__version__.split(".")[0]) >= 5
        if v5 and llm_name in TOKENIZER_COMPAT_V5:
            tk.update(TOKENIZER_COMPAT_V5[llm_name])
            self.load_info["compat_shims"].append(f"tokenizer {TOKENIZER_COMPAT_V5[llm_name]}")
        self.config = self._config(llm_name, token, revision, trust_remote_code)
        if self.config is not None:
            tk["config"] = self.config
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, **tk)
        assert not isinstance(self.tokenizer, bool), "tokenizer load returned a bool (ENV.md item 2)"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.with_system_prompt = True
        if chat_template_fallback is not None:
            assert getattr(self.tokenizer, "chat_template", None) is None
            self.tokenizer.chat_template = chat_template_fallback
            self.load_info["chat_template_source"] = "D006 fallback (v1 only)"
        else:
            self.load_info["chat_template_source"] = (
                "model's own" if getattr(self.tokenizer, "chat_template", None) else "NONE")
        self.load_info["tokenizer_class"] = type(self.tokenizer).__name__
        self.supports_system_role = has_system_role(self.tokenizer)
        self.model = None
        if tokenizer_only:
            return
        mk = dict(token=token, revision=revision, device_map="cuda")
        mk["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = torch.bfloat16
        if trust_remote_code:
            mk["trust_remote_code"] = True
        if self.config is not None:
            mk["config"] = self.config
        try:
            self.model = AutoModelForCausalLM.from_pretrained(llm_name, **mk)
            self.load_info["model_class_route"] = "AutoModelForCausalLM"
        except (ValueError, KeyError) as e:
            arch = AutoConfig.from_pretrained(llm_name, token=token, revision=revision).architectures[0]
            self.model = getattr(transformers, arch).from_pretrained(llm_name, **mk)
            self.load_info["model_class_route"] = f"architectures[0] ({str(e)[:120]})"
        self.load_info["model_class"] = type(self.model).__name__
        self.model.eval()
        gc = self.model.generation_config
        self.load_info["use_cache_overridden"] = False
        if getattr(gc, "use_cache", True) is False:
            gc.use_cache = True
            self.load_info["use_cache_overridden"] = True
        self.load_info["hf_revision"] = getattr(self.model.config, "_commit_hash", None)
        self.load_info["generation_config"] = {k: v for k, v in gc.to_dict().items()
                                               if k in ("do_sample", "temperature", "top_p", "top_k",
                                                        "repetition_penalty", "eos_token_id", "max_length")}

    def _config(self, llm_name, token, revision, trust_remote_code):
        """None if the stock config loads; else the LongRoPE-patched config."""
        try:
            AutoConfig.from_pretrained(llm_name, token=token, revision=revision,
                                       trust_remote_code=trust_remote_code)
            return None
        except KeyError as e:
            if "original_max_position_embeddings" not in str(e):
                raise
        import json
        from huggingface_hub import hf_hub_download
        d = json.load(open(hf_hub_download(llm_name, "config.json", revision=revision, token=token)))
        d["rope_scaling"] = dict(d["rope_scaling"],
                                 original_max_position_embeddings=d["original_max_position_embeddings"])
        cfg = AutoConfig.for_model(d.pop("model_type"), **{k: v for k, v in d.items() if k != "auto_map"})
        self.load_info["compat_shims"].append(
            f"config: rope_scaling.original_max_position_embeddings={d['original_max_position_embeddings']} (copied from top level)")
        return cfg

    def make_prompt(self, system, user):
        return render(self.tokenizer, self.supports_system_role, system, user)
