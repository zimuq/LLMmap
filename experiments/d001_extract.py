"""
D001 / P1 — S1 (feature extraction) + S2 (template provenance check).

The only expensive stage. Extracts [CLS] features for the test split against
the shipped templates, caches everything to .npy so all downstream statistics
(S3-S6) are cheap re-runs.

Run from repo root with PYTHONPATH=. — see experiments/submit_d001.slurm.
"""
import os
import json
import random
import argparse

import numpy as np
import torch
from scipy.spatial.distance import cdist

from LLMmap.inference import load_LLMmap
from LLMmap.dataset import load_datasets
from LLMmap.templates import infer_features, build_templates

SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./data/pretrained_models/default")
    ap.add_argument("--out", default="./results/D001")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # read_dataset (dataset_maker.py:8) shuffles with the global RNG and no seed;
    # seed it so runs are bit-reproducible.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[S1] loading LLMmap...", flush=True)
    conf, inf = load_LLMmap(args.model_dir, device=args.device)
    if not conf["is_open"]:
        raise SystemExit("D001 requires the open-set model.")

    # Capture the template/label order BEFORE load_datasets mutates conf in place
    # (dataset.py:190-195 overwrites llms_map / queries / num_classes).
    inf_label_order = [inf.label_map[i] for i in range(len(inf.label_map))]
    DB = np.asarray(inf.DB)  # (52, 384), rows in sorted(templates_map) order
    print(f"[S1] DB shape={DB.shape}", flush=True)

    print("[S1] embedding corpus (dominant cost)...", flush=True)
    (loader_train, loader_test), cache, _ = load_datasets(conf, siamese=False)

    # --- GUARD (P1 S1): label index order must agree, else every downstream
    # number is silently wrong. Per plan: stop, do not patch around it.
    cache_order = [k for k, _ in sorted(cache.llms_map.items(), key=lambda kv: kv[1])]
    if cache_order != inf_label_order:
        raise SystemExit(
            "FATAL: label order mismatch between cache.llms_map and inf.label_map.\n"
            f"  cache[:3]={cache_order[:3]}\n  inf[:3]={inf_label_order[:3]}"
        )
    print(f"[S1] label-order guard OK ({len(cache_order)} models)", flush=True)

    print("[S1] extracting test features...", flush=True)
    y_test, f_test = infer_features(inf.model, loader_test, args.device)
    print(f"[S1] f_test={f_test.shape} y_test={y_test.shape}", flush=True)

    print("[S1] extracting train features (for S2 provenance)...", flush=True)
    y_train, f_train = infer_features(inf.model, loader_train, args.device)
    print(f"[S1] f_train={f_train.shape}", flush=True)

    # Distance from every test trace to every shipped template.
    D = cdist(f_test, DB, metric=conf.get("distance_fn", "euclidean"))
    print(f"[S1] D={D.shape}", flush=True)

    # --- S2: template provenance. The repo records no provenance for the
    # shipped templates.json; recompute from the train split and compare.
    tmpl_recomputed = build_templates(y_train, f_train)
    if tmpl_recomputed.shape == DB.shape:
        absdev = np.abs(tmpl_recomputed - DB)
        prov = {
            "max_abs_dev": float(absdev.max()),
            "mean_abs_dev": float(absdev.mean()),
            "template_scale_median_l2": float(np.linalg.norm(DB, axis=1).mean()),
            "shape_match": True,
        }
    else:
        prov = {"shape_match": False,
                "recomputed_shape": list(tmpl_recomputed.shape),
                "shipped_shape": list(DB.shape)}
    print(f"[S2] provenance: {prov}", flush=True)

    np.save(f"{args.out}/f_test.npy", f_test)
    np.save(f"{args.out}/y_test.npy", y_test)
    np.save(f"{args.out}/D.npy", D)
    np.save(f"{args.out}/DB.npy", DB)
    np.save(f"{args.out}/templates_recomputed.npy", tmpl_recomputed)
    with open(f"{args.out}/labels.json", "w") as fh:
        json.dump({"label_order": inf_label_order, "seed": SEED}, fh, indent=2)
    with open(f"{args.out}/provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)

    print("[S1/S2] DONE", flush=True)


if __name__ == "__main__":
    main()
