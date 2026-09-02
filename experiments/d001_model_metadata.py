"""
D001 / M3 — structural annotation of the 52 shipped models.

Per D001 Constraints: "Ground truth for 'is X a fine-tune of Y' comes from model
cards and naming. Flag any pair where the relationship is inferred rather than
documented."

Fields per model:
  org        organisation
  lineage    broad model line; two models sharing a lineage are same-family
  variant    distinguishing tag within a lineage (used for adjacent-version)
  base       canonical name of the model this is a fine-tune / derivative OF,
             "" if it is itself a base-line instruct model
  base_prov  "documented" (stated on the model card / paper) | "inferred"
  arch       architecture family
  tok        tokenizer family
  params_b   parameter count in billions; for MoE this is TOTAL params
             (per D001 Review: total is the correct basis, memory residency)
  active_b   active params for MoE (None for dense) -- sensitivity only
  proprietary  size not inferable / closed API

arch and tok are best-effort from public specs, not verified against configs
here; they are secondary descriptive columns and do not carry the verdict.
The base / lineage / variant columns do carry it and are annotated carefully.
"""

# name -> dict
MODELS = {
    # --- Cohere -------------------------------------------------------------
    "CohereForAI/aya-23-35B": dict(org="cohere", lineage="aya-23", variant="35b", base="", base_prov="", arch="cohere", tok="cohere", params_b=35.0),
    "CohereForAI/aya-23-8B": dict(org="cohere", lineage="aya-23", variant="8b", base="", base_prov="", arch="cohere", tok="cohere", params_b=8.0),

    # --- Deci ---------------------------------------------------------------
    "Deci/DeciLM-7B-instruct": dict(org="deci", lineage="decilm", variant="7b", base="", base_prov="", arch="decilm", tok="llama2", params_b=7.0),

    # --- Mistral line (base) and its derivatives ----------------------------
    "mistralai/Mistral-7B-Instruct-v0.1": dict(org="mistralai", lineage="mistral-7b", variant="v0.1", base="", base_prov="", arch="mistral", tok="mistral", params_b=7.0),
    "mistralai/Mistral-7B-Instruct-v0.2": dict(org="mistralai", lineage="mistral-7b", variant="v0.2", base="", base_prov="", arch="mistral", tok="mistral", params_b=7.0),
    "mistralai/Mistral-7B-Instruct-v0.3": dict(org="mistralai", lineage="mistral-7b", variant="v0.3", base="", base_prov="", arch="mistral", tok="mistral", params_b=7.0),
    # zephyr-7b-beta model card: fine-tuned from mistralai/Mistral-7B-v0.1
    "HuggingFaceH4/zephyr-7b-beta": dict(org="huggingfaceh4", lineage="mistral-7b", variant="zephyr-beta", base="mistralai/Mistral-7B-Instruct-v0.1", base_prov="documented", arch="mistral", tok="mistral", params_b=7.0),
    # openchat_3.5 model card: based on Mistral-7B
    "openchat/openchat_3.5": dict(org="openchat", lineage="mistral-7b", variant="oc3.5", base="mistralai/Mistral-7B-Instruct-v0.1", base_prov="documented", arch="mistral", tok="mistral", params_b=7.0),
    # SOLAR: depth up-scaling FROM Mistral-7B weights (SOLAR paper). Not a plain
    # fine-tune -- layer-duplicated then continued-pretrained. Marked inferred.
    "upstage/SOLAR-10.7B-Instruct-v1.0": dict(org="upstage", lineage="mistral-7b", variant="solar-10.7b", base="mistralai/Mistral-7B-Instruct-v0.1", base_prov="inferred", arch="llama", tok="mistral", params_b=10.7),

    # --- Mixtral ------------------------------------------------------------
    "mistralai/Mixtral-8x7B-Instruct-v0.1": dict(org="mistralai", lineage="mixtral-8x7b", variant="v0.1", base="", base_prov="", arch="mixtral-moe", tok="mistral", params_b=46.7, active_b=12.9),
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": dict(org="nousresearch", lineage="mixtral-8x7b", variant="hermes2-dpo", base="mistralai/Mixtral-8x7B-Instruct-v0.1", base_prov="documented", arch="mixtral-moe", tok="mistral", params_b=46.7, active_b=12.9),

    # --- Qwen ---------------------------------------------------------------
    "Qwen/Qwen2-1.5B-Instruct": dict(org="qwen", lineage="qwen2", variant="2-1.5b", base="", base_prov="", arch="qwen2", tok="qwen", params_b=1.5),
    "Qwen/Qwen2-7B-Instruct": dict(org="qwen", lineage="qwen2", variant="2-7b", base="", base_prov="", arch="qwen2", tok="qwen", params_b=7.0),
    "Qwen/Qwen2-72B-Instruct": dict(org="qwen", lineage="qwen2", variant="2-72b", base="", base_prov="", arch="qwen2", tok="qwen", params_b=72.0),
    "Qwen/Qwen2.5-0.5B-Instruct": dict(org="qwen", lineage="qwen2", variant="2.5-0.5b", base="", base_prov="", arch="qwen2", tok="qwen", params_b=0.5),
    "Qwen/Qwen2.5-3B-Instruct": dict(org="qwen", lineage="qwen2", variant="2.5-3b", base="", base_prov="", arch="qwen2", tok="qwen", params_b=3.0),

    # --- Llama 2 ------------------------------------------------------------
    "meta-llama/Llama-2-7b-chat-hf": dict(org="meta", lineage="llama-2", variant="7b-chat", base="", base_prov="", arch="llama", tok="llama2", params_b=7.0),
    # Together's 32K context extension of Llama-2-7B (model card)
    "togethercomputer/Llama-2-7B-32K-Instruct": dict(org="togethercomputer", lineage="llama-2", variant="7b-32k", base="meta-llama/Llama-2-7b-chat-hf", base_prov="documented", arch="llama", tok="llama2", params_b=7.0),

    # --- Llama 3 / 3.1 / 3.2 and derivatives --------------------------------
    "meta-llama/Meta-Llama-3-8B-Instruct": dict(org="meta", lineage="llama-3", variant="3-8b", base="", base_prov="", arch="llama", tok="llama3", params_b=8.0),
    "meta-llama/Meta-Llama-3-70B-Instruct": dict(org="meta", lineage="llama-3", variant="3-70b", base="", base_prov="", arch="llama", tok="llama3", params_b=70.0),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": dict(org="meta", lineage="llama-3", variant="3.1-8b", base="", base_prov="", arch="llama", tok="llama3", params_b=8.0),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": dict(org="meta", lineage="llama-3", variant="3.1-70b", base="", base_prov="", arch="llama", tok="llama3", params_b=70.0),
    "meta-llama/Llama-3.2-1B-Instruct": dict(org="meta", lineage="llama-3", variant="3.2-1b", base="", base_prov="", arch="llama", tok="llama3", params_b=1.0),
    "meta-llama/Llama-3.2-3B-Instruct": dict(org="meta", lineage="llama-3", variant="3.2-3b", base="", base_prov="", arch="llama", tok="llama3", params_b=3.0),
    # Abacus.AI Smaug: fine-tune of Meta-Llama-3-70B-Instruct (model card).
    # THIS IS D001's MOTIVATING CASE (M4).
    "abacusai/Smaug-Llama-3-70B-Instruct": dict(org="abacusai", lineage="llama-3", variant="smaug-70b", base="meta-llama/Meta-Llama-3-70B-Instruct", base_prov="documented", arch="llama", tok="llama3", params_b=70.0),
    # Gradient: context-extended Llama-3-8B-Instruct (model card)
    "gradientai/Llama-3-8B-Instruct-Gradient-1048k": dict(org="gradientai", lineage="llama-3", variant="grad-1048k", base="meta-llama/Meta-Llama-3-8B-Instruct", base_prov="documented", arch="llama", tok="llama3", params_b=8.0),
    # NVIDIA ChatQA-1.5 built on Llama-3-8B (model card)
    "nvidia/Llama3-ChatQA-1.5-8B": dict(org="nvidia", lineage="llama-3", variant="chatqa-1.5", base="meta-llama/Meta-Llama-3-8B-Instruct", base_prov="documented", arch="llama", tok="llama3", params_b=8.0),
    # OpenChat 3.6 is Llama-3-8B based (model card)
    "openchat/openchat-3.6-8b-20240522": dict(org="openchat", lineage="llama-3", variant="oc3.6-8b", base="meta-llama/Meta-Llama-3-8B-Instruct", base_prov="documented", arch="llama", tok="llama3", params_b=8.0),

    # --- Google Gemma -------------------------------------------------------
    "google/gemma-2b-it": dict(org="google", lineage="gemma", variant="1.0-2b", base="", base_prov="", arch="gemma", tok="gemma", params_b=2.0),
    "google/gemma-7b-it": dict(org="google", lineage="gemma", variant="1.0-7b", base="", base_prov="", arch="gemma", tok="gemma", params_b=7.0),
    "google/gemma-1.1-2b-it": dict(org="google", lineage="gemma", variant="1.1-2b", base="", base_prov="", arch="gemma", tok="gemma", params_b=2.0),
    "google/gemma-1.1-7b-it": dict(org="google", lineage="gemma", variant="1.1-7b", base="", base_prov="", arch="gemma", tok="gemma", params_b=7.0),
    "google/gemma-2-9b-it": dict(org="google", lineage="gemma", variant="2-9b", base="", base_prov="", arch="gemma2", tok="gemma", params_b=9.0),
    "google/gemma-2-27b-it": dict(org="google", lineage="gemma", variant="2-27b", base="", base_prov="", arch="gemma2", tok="gemma", params_b=27.0),

    # --- Microsoft Phi ------------------------------------------------------
    "microsoft/Phi-3-mini-4k-instruct": dict(org="microsoft", lineage="phi-3", variant="mini-4k", base="", base_prov="", arch="phi3", tok="llama2", params_b=3.8),
    "microsoft/Phi-3-mini-128k-instruct": dict(org="microsoft", lineage="phi-3", variant="mini-128k", base="", base_prov="", arch="phi3", tok="llama2", params_b=3.8),
    "microsoft/Phi-3.5-mini-instruct": dict(org="microsoft", lineage="phi-3", variant="3.5-mini", base="", base_prov="", arch="phi3", tok="llama2", params_b=3.8),
    "microsoft/Phi-3-medium-4k-instruct": dict(org="microsoft", lineage="phi-3", variant="medium-4k", base="", base_prov="", arch="phi3", tok="llama2", params_b=14.0),
    "microsoft/Phi-3-medium-128k-instruct": dict(org="microsoft", lineage="phi-3", variant="medium-128k", base="", base_prov="", arch="phi3", tok="llama2", params_b=14.0),
    "microsoft/Phi-3.5-MoE-instruct": dict(org="microsoft", lineage="phi-3", variant="3.5-moe", base="", base_prov="", arch="phi3-moe", tok="llama2", params_b=41.9, active_b=6.6),

    # --- IBM Granite --------------------------------------------------------
    "ibm-granite/granite-3.0-8b-instruct": dict(org="ibm", lineage="granite-3", variant="3.0-8b", base="", base_prov="", arch="granite", tok="granite", params_b=8.0),
    "ibm-granite/granite-3.1-8b-instruct": dict(org="ibm", lineage="granite-3", variant="3.1-8b", base="", base_prov="", arch="granite", tok="granite", params_b=8.0),

    # --- TII Falcon 3 -------------------------------------------------------
    "tiiuae/Falcon3-7B-Instruct": dict(org="tii", lineage="falcon-3", variant="7b", base="", base_prov="", arch="falcon3", tok="falcon3", params_b=7.0),
    "tiiuae/Falcon3-10B-Instruct": dict(org="tii", lineage="falcon-3", variant="10b", base="", base_prov="", arch="falcon3", tok="falcon3", params_b=10.0),

    # --- others -------------------------------------------------------------
    "internlm/internlm2_5-7b-chat": dict(org="internlm", lineage="internlm2", variant="2.5-7b", base="", base_prov="", arch="internlm2", tok="internlm", params_b=7.0),
    "utter-project/EuroLLM-1.7B-Instruct": dict(org="utter", lineage="eurollm", variant="1.7b", base="", base_prov="", arch="llama", tok="eurollm", params_b=1.7),

    # --- proprietary (size not inferable) -----------------------------------
    "gpt-3.5-turbo": dict(org="openai", lineage="gpt", variant="3.5-turbo", base="", base_prov="", arch="proprietary", tok="tiktoken", params_b=None, proprietary=True),
    "gpt-4-turbo-2024-04-09": dict(org="openai", lineage="gpt", variant="4-turbo", base="", base_prov="", arch="proprietary", tok="tiktoken", params_b=None, proprietary=True),
    "gpt-4o-2024-05-13": dict(org="openai", lineage="gpt", variant="4o", base="", base_prov="", arch="proprietary", tok="tiktoken", params_b=None, proprietary=True),
    "claude-3-haiku-20240307": dict(org="anthropic", lineage="claude-3", variant="haiku", base="", base_prov="", arch="proprietary", tok="claude", params_b=None, proprietary=True),
    "claude-3-opus-20240229": dict(org="anthropic", lineage="claude-3", variant="opus", base="", base_prov="", arch="proprietary", tok="claude", params_b=None, proprietary=True),
    "claude-3-5-sonnet-20240620": dict(org="anthropic", lineage="claude-3", variant="3.5-sonnet", base="", base_prov="", arch="proprietary", tok="claude", params_b=None, proprietary=True),
}

for _m in MODELS.values():
    _m.setdefault("active_b", None)
    _m.setdefault("proprietary", False)


def pair_labels(a: str, b: str) -> dict:
    """Structural relationship labels for an unordered pair."""
    ma, mb = MODELS[a], MODELS[b]

    # one is a fine-tune of the other, or both derive from the same base
    same_base = (ma["base"] == b) or (mb["base"] == a) or \
                (ma["base"] != "" and ma["base"] == mb["base"])
    same_lineage = ma["lineage"] == mb["lineage"]
    # adjacent version: same lineage, neither is a derivative of the other,
    # different variant tag
    adjacent_version = same_lineage and not same_base and ma["variant"] != mb["variant"]

    inferred = (same_base and "inferred" in (ma["base_prov"], mb["base_prov"]))

    return dict(
        same_base=same_base,
        adjacent_version=adjacent_version,
        same_family=same_lineage,
        same_org=ma["org"] == mb["org"],
        same_arch=ma["arch"] == mb["arch"],
        same_tokenizer=ma["tok"] == mb["tok"],
        # I1's definition: "base + fine-tune, or adjacent versions in one family"
        near_relative=same_base or adjacent_version,
        # broader union used for the amended primary test (power)
        structurally_related=same_base or adjacent_version or same_lineage,
        relation_inferred=inferred,
    )


def le_14b(name: str, include_boundary: bool = True) -> bool:
    """<=14B under TOTAL-parameter accounting (D001 Review: total is correct).

    include_boundary=True  -> Phi-3-medium (exactly 14B) INCLUDED  (37/9/6)
    include_boundary=False -> strict <14B, Phi-3-medium excluded   (35/11/6)
    """
    p = MODELS[name]["params_b"]
    if p is None:
        return False  # proprietary: size not inferable, excluded from the subset
    return p <= 14.0 if include_boundary else p < 14.0
