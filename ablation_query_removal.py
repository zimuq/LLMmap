"""
Ablation: does perturbing alignment-family query responses tank
fingerprinting accuracy, as claimed in the LLMmap paper (Sec. 8.1,
"blocking only two classes of queries reduces the fingerprint accuracy
by more than 50%")?

Correction vs. earlier version of this script: InferenceModel.__call__
(LLMmap/inference.py:105-107) hard-requires len(answers) == len(queries)
(always 8) -- it concatenates the model's precomputed embeddings of the
8 fixed queries with the embeddings of the 8 given answers positionally
(torch.cat(dim=1)), so shrinking the answer list either crashes (row
count mismatch) or silently misaligns query/answer pairs. Dropping
queries is not a valid way to simulate "blocking" them.

This matches the paper's own threat model better anyway (Sec. 8.1): the
defender doesn't remove queries, they intercept flagged categories
(banner-grabbing / alignment-error responses) and substitute a fixed
perturbed response before returning it to the attacker -- "Fixed
Response" mitigation: return the string "I cannot answer that." So here
we *substitute* a placeholder answer at the targeted indices, keeping
all 8 slots intact.

Query categories (paper Sec. 4), 0-indexed against confs/queries/default.json:
  0: banner-grabbing + prompt-injection   ("...who created you")
  1: alignment - harmful refusal           ("Tell me how to build a bomb.")
  2: meta-information                      ("size of your training dataset")
  3: alignment - weak/controversial        ("true-or-false: climate change...")
  4: meta-information                      ("data cutoff date")
  5: banner-grabbing + prompt-injection    ("...What model are you exactly?")
  6: banner-grabbing (plain)               ("What LLM are you exactly?")
  7: alignment - weak/controversial + inj. ("...is 9/11 real?")
"""
import argparse
import itertools
import numpy as np
import tqdm

from LLMmap.dataset import read_dataset
from LLMmap.inference import load_LLMmap

DEFAULT_PLACEHOLDER = "I cannot answer that."


def evaluate_topk_perturbed(model, test_iterable, perturb_idx, placeholder, k_values=(1, 2, 3)):
    num_samples = 0
    topk_correct = {k: 0 for k in k_values}
    llms_map = {v: k for k, v in model.label_map.items()}

    for entry in tqdm.tqdm(test_iterable):
        gt_label = llms_map[entry['llm']]
        answers = [trace[1] for trace in entry['traces']]
        for i in perturb_idx:
            answers[i] = placeholder

        distances = model(answers)
        num_samples += 1

        for k in k_values:
            topk_idx = np.argsort(distances)[:k]
            if gt_label in topk_idx:
                topk_correct[k] += 1

    return {k: topk_correct[k] / num_samples for k in k_values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_home_dir")
    parser.add_argument("--perturb", type=int, nargs="*", default=[],
                         help="0-indexed query positions whose answer gets replaced with --placeholder")
    parser.add_argument("--placeholder", type=str, default=DEFAULT_PLACEHOLDER,
                         help="Fixed response substituted at perturbed positions (paper Sec. 8.1)")
    parser.add_argument("-k", "--topk", type=int, default=3)
    parser.add_argument("-m", "--max-entries", type=int, default=None)
    args = parser.parse_args()

    conf, inf = load_LLMmap(args.model_home_dir, device='cpu')
    if not conf['is_open']:
        raise SystemExit("Applicable to only open-set inference model.")

    train, test = read_dataset(conf['dataset_path'])
    test_iter = test if args.max_entries is None else list(itertools.islice(test, args.max_entries))

    num_q = len(conf['queries'])
    print(f"Total queries in strategy: {num_q}. Perturbing positions {args.perturb} with {args.placeholder!r}:")
    for i in range(num_q):
        tag = "PERTURBED" if i in args.perturb else "kept"
        print(f"  [{i}] ({tag}) {conf['queries'][i][:70]!r}")

    k_values = tuple(range(1, args.topk + 1))
    print("Running ablation...")
    acc = evaluate_topk_perturbed(inf, test_iter, args.perturb, args.placeholder, k_values=k_values)
    for k in k_values:
        print(f"Top-{k} accuracy: {acc[k]:.3%}")
