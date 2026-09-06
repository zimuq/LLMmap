import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

TRAIN, TEST = 'train', 'test'
# D006/S3 (C4): the three-way config split. TEST is shared with the legacy
# two-pool naming above -- same string, same meaning. TRAIN remains only for
# reproducing D001/D004/D005 against the legacy `train_test_split.json`.
BUILD, VAL = 'build', 'val'

def sample_from_multi_universe(universe):
    sample = {}
    for k, u in universe.items():
        sample[k] = random.sample(u, 1)[0]
    return sample

###############################################################################
# Data classes                                                               #
###############################################################################

class PromptConf:
    """A concrete prompt + decoding‑parameters bundle.

    Calling a *PromptConf* with a *query* returns a ready‑to‑feed prompt string
    and the corresponding sampling hyper‑parameters.
    """

    def __init__(
        self,
        sampling_hparams: Dict[str, Any],
        system_prompt: Optional[str],
        cot_prompt: Optional[str] = None,
        rag_prompt: Optional[str] = None,
        raw: Sequence[Any] | None = None,
    ) -> None:
        self.sampling_hparams = sampling_hparams
        self.system_prompt = system_prompt or ""
        self.cot_prompt = cot_prompt
        self.rag_prompt = rag_prompt
        self.raw = raw or []

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def __call__(self, query: str, llm, apply_template: bool = True):
        """Materialise the prompt and return *(prompt, sampling_hparams)*."""
        # Chain‑of‑thought augmentation ------------------------------------------------
        if self.cot_prompt:
            query = self.cot_prompt % query

        # Retrieval‑augmented generation augmentation ---------------------------------
        if self.rag_prompt:
            query = self.rag_prompt % query

        # Final assembly --------------------------------------------------------------
        if apply_template:
            prompt_str = llm.make_prompt(self.system_prompt, query)
        else:
            prompt_str = (query, self.system_prompt)

        return prompt_str, self.sampling_hparams

    # ---------------------------------------------------------------------

    def __str__(self) -> str:
        raw_str = " ".join(map(str, self.raw))
        return raw_str

    # ---------------------------------------------------------------------
    # Identity (D006/F3, Call 4 approved 2026-09-05)
    #
    # `PromptConfFactory.sample()` deduplicates through a `set()` and its
    # docstring promises "n unique confs". Without __eq__/__hash__ Python falls
    # back to identity hashing, so no freshly-constructed object was ever
    # rejected and the dedup silently did nothing: measured over 40 seeds, a
    # 75-config S_build carried a mean of 3.95 duplicate configurations, a third
    # of them in the sparsest cell (no system prompt, no CoT, no RAG).
    #
    # The key is `(raw, sampling_hparams)` -- the *design point*. Deliberately
    # NOT the materialised `rag_prompt`: that string embeds a randomly drawn
    # document (`_generate_rag_prompt`), so keying on it would practically never
    # collide and would leave the defect in place under a new name.
    # ---------------------------------------------------------------------

    def _signature(self) -> str:
        # json rather than tuple(): `raw` holds rag templates, which are lists.
        return json.dumps([self.raw, sorted(self.sampling_hparams.items())],
                          sort_keys=True, default=str)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PromptConf):
            return NotImplemented
        return self._signature() == other._signature()

    def __hash__(self) -> int:
        return hash(self._signature())


    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampling_hparams": self.sampling_hparams,
            "system_prompt": self.system_prompt,
            "cot_prompt": self.cot_prompt,
            "rag_prompt": self.rag_prompt,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptConf":
        return cls(
            sampling_hparams=data.get("sampling_hparams", {}),
            system_prompt=data.get("system_prompt", ""),
            cot_prompt=data.get("cot_prompt"),
            rag_prompt=data.get("rag_prompt"),
            raw=data.get("raw", []),
        )

###############################################################################
# JSON‑driven factory                                                         #
###############################################################################

class _ConfigLoader:
    """Utility class that lazily loads JSON config files from disk.

    Attributes from *general.json* act as a global fallback whenever the
    dedicated file is missing or a key cannot be resolved.
    """

    def __init__(self, home_dir: Union[str, Path]):
        self._root = Path(home_dir).expanduser().resolve()
        if not self._root.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self._root}")

        # *general.json* is mandatory because it holds fallbacks & constants.
        self._general: Dict[str, Any] = self._read_json("general.json", required=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, filename: str, fallback_key: str, default: Any) -> Any:
        """Return JSON content or a fallback from *general.json*.

        If *filename* is missing or returns an empty structure, the value of
        *fallback_key* inside *general.json* is returned instead. If that key
        is also absent, *default* is returned.
        """
        data = self._read_json(filename, required=False)
        if data:
            return data
        return self._general.get(fallback_key, default)

    def constant(self, key: str, default: Any = None) -> Any:
        """Read a scalar constant from *general.json* (with optional default)."""
        return self._general.get(key, default)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_json(self, filename: str, *, required: bool) -> Any:

        path = self._root / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required configuration file missing: {path}")
            return None

        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        return data

###############################################################################
# The main factory                                                            #
###############################################################################

class PromptConfFactory:
    """Sample *PromptConf* objects based on JSON configuration files.

    Parameters
    ----------
    home_dir:
        Path to the project root. The actual JSON files are expected under –
        ``{home_dir}/confs/prompt_configurations``.
    """

    def __init__(self, home_dir: Union[str, Path]):
        self._cfg = _ConfigLoader(home_dir)

        # Collections ------------------------------------------------------
        self.sampling_universe: Dict[str, Any] = self._cfg.constant("sampling_universe", {})
        # D005/M2: per-pool value sets for sampling hyper-parameters, so I2 holds
        # for them as it already does for prompt collections. Any parameter absent
        # from this map is drawn from the full universe in BOTH pools -- see
        # `sampling_universe_shared` in general.json and DECISIONS.md A5.
        self.sampling_universe_split: Dict[str, Any] = self._cfg.constant(
            "sampling_universe_split", {})
        self.sampling_universe_shared: List[str] = self._cfg.constant(
            "sampling_universe_shared", [])

        self.params = {
            'systems' :  self._cfg.load("systems.json", "system_prompts", []),
            'cot_prompts' : self._cfg.load("cot_prompts.json", "cot_prompts", []),
            'rag_prompts' : self._cfg.load("rag_prompts.json", "rag_templates", []),
        }
       
        self.documents_rag: List[Tuple[Any, Any, List[str]]] = self._cfg.load("rag_context.json", "documents_rag", [])

        # D006/S3: prefer the three-way build/val/test split (C4, `split_v2.json`)
        # and fall back to the legacy two-pool `train_test_split.json`. The legacy
        # file is deliberately NOT overwritten -- it is the artifact D005/M4's
        # forensics reasoned about and what LLMmap's shipped corpus was generated
        # under, so D001/D004/D005 stay reproducible against it.
        _v2 = self._cfg.load("split_v2.json", "__absent__", None)
        self.split_schema_version: str = "cdqd-split-v1-legacy"
        if _v2:
            self.split_schema_version = _v2.get("schema_version", "cdqd-split-v2")
            self.train_test_split = {k: v for k, v in _v2.items()
                                     if k in (BUILD, VAL, TEST)}
        else:
            self.train_test_split = self._cfg.load(
                "train_test_split.json", "train_test_split", {})
        
        # Scalars / probabilities -----------------------------------------
        self.COT_P: float = self._cfg.constant("COT_P", 0.0)
        self.RAG_P: float = self._cfg.constant("RAG_P", 0.0)
        self.MIN_CHUNKS_RAG: int = self._cfg.constant("MIN_CHUNKS_RAG", 1)
        self.MAX_CHUNKS_RAG: int = self._cfg.constant("MAX_CHUNKS_RAG", 2)

        self.WITH_SYSTEM_P: int = self._cfg.constant("WITH_SYSTEM_P", 1)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _generate_rag_prompt(self, rag_template: Tuple[str, str]) -> Optional[str]:
        t_body, t_chunk = rag_template

        # Pick a random document from the retrieval corpus --------------
        if not self.documents_rag:
            return None

        _, _, background_texts = random.choice(self.documents_rag)
        random.shuffle(background_texts)

        n_chunks = random.randint(self.MIN_CHUNKS_RAG, self.MAX_CHUNKS_RAG)
        chunks = background_texts[:n_chunks]
        chunks_text = "".join(t_chunk % c for c in chunks)

        # Guard: template placeholders should be fully resolved ---------
        if "%" in chunks_text:
            return None

        return t_body.format(retrieved_chunk=chunks_text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _sample_hparams(self, pool=TRAIN) -> Dict[str, Any]:
        """Draw sampling hyper-parameters from `pool`'s value set (D005/M2).

        A parameter listed in `sampling_universe_shared` is drawn from the full
        universe in both pools. That is a deliberate, documented carve-out from
        I2, not an oversight: `do_sample` has only two distinct values, so any
        disjoint split would put all-greedy decoding in one pool and
        all-stochastic in the other -- a worse confound than the leakage it
        removes. See DECISIONS.md A5 (resolved 2026-09-02, option 1).
        """
        # D006/S3: fail LOUDLY on an unknown pool. Previously an unrecognised
        # pool name silently yielded `{}` here and every parameter fell back to
        # the FULL universe -- i.e. a silent I2 violation, the exact failure mode
        # D005 existed to fix, reintroduced by the two-pool -> three-pool rename
        # (`train` is not a key in split_v2). `_cond_choice` already raises
        # KeyError for the same mistake; this makes the two consistent.
        if self.sampling_universe_split and pool not in self.sampling_universe_split:
            raise KeyError(
                f"pool '{pool}' is not in sampling_universe_split "
                f"(have: {sorted(self.sampling_universe_split)}). Refusing to "
                f"fall back to the full universe -- that would silently break I2.")
        per_pool = self.sampling_universe_split.get(pool, {})
        out: Dict[str, Any] = {}
        for k, universe in self.sampling_universe.items():
            values = universe if k in self.sampling_universe_shared \
                else per_pool.get(k, universe)
            out[k] = random.sample(list(values), 1)[0]
        return out

    def sample_one(self, pool=TRAIN) -> PromptConf:
        """Return a freshly sampled *PromptConf* instance."""
        sampling_hparams = self._sample_hparams(pool)
        
        system_prompt = self._cond_choice("systems", self.WITH_SYSTEM_P, pool)
        cot_prompt = self._cond_choice("cot_prompts", self.COT_P, pool)
        rag_template = self._cond_choice("rag_prompts", self.RAG_P, pool)

        rag_prompt = self._generate_rag_prompt(rag_template) if rag_template else None

        raw = (system_prompt, cot_prompt, rag_template)
        return PromptConf(
            sampling_hparams=sampling_hparams,
            system_prompt=system_prompt,
            cot_prompt=cot_prompt,
            rag_prompt=rag_prompt,
            raw=raw,
        )
    
    def _cond_choice(self, collection_name, p, pool):
        collection = self.params[collection_name]
        avaliable = self.train_test_split[pool][collection_name]
        idx = random.choice(avaliable) if avaliable and random.random() < p else None
        return None if idx is None else collection[idx]

    def sample(self, n, pool=TRAIN):
        """Sample n unique confs from `pool`.

        D005/M1: `pool` was accepted here but never forwarded to `sample_one`,
        so every caller -- including dataset_maker.py's `pool=TEST` call -- drew
        from the TRAIN pool. That defeated the train/test holdout silently, with
        no error, which is invariant I2's stated failure mode.

        D006/F3: dedup here is only real because `PromptConf` now defines
        `__eq__`/`__hash__` (see its `_signature`). Before that, identity
        hashing made this loop a no-op. The retry cap exists because with a
        working `set()` this loop CAN now spin forever if `pool`'s config space
        is smaller than `n` -- which the identity-hashing version could never
        do. Failing loudly beats hanging a 37-shard job.

        The cap counts *consecutive* draws that add nothing, not total attempts:
        a total-attempt budget has to scale with `n` to avoid false failures,
        which makes it uselessly large exactly when `n` is big. Consecutive
        misses measure what actually matters -- that the pool is exhausted --
        independently of `n`.
        """
        assert n > 0
        s, misses, max_misses = set(), 0, 5000
        while len(s) != n:
            before = len(s)
            s.add(self.sample_one(pool))
            misses = 0 if len(s) > before else misses + 1
            if misses >= max_misses:
                raise RuntimeError(
                    f"could not draw {n} distinct configs from pool '{pool}': "
                    f"{max_misses} consecutive draws produced no new config "
                    f"(got {len(s)}). The pool's config space is smaller than n.")
        return list(s)
            

