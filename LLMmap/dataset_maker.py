import tqdm
import json
import random 

from .llm import load_llm
from .prompt_configuration import TRAIN, TEST

def read_dataset(
    path,
    encoding='utf-8',
    shuffle=True
):
    train, test = [], []
    
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            entry = json.loads(line)
            if entry['dataset'] == TRAIN:
                dest = train
            elif entry['dataset'] == TEST:
                dest = test

            entry.pop('dataset')
            dest.append(entry)
            
    if shuffle:
        random.shuffle(train)
        random.shuffle(test)
    return train, test


GEN_BATCH_SIZE = 8


def make_dataset_entries_for_new_llm(
    llm,
    queries,
    prompt_confs,
    pool=TRAIN,
    batch_size=GEN_BATCH_SIZE,
    max_new_tokens=None,
):
    """Generate one entry per prompt-config, batching queries within a config.

    D006/S1 (from D002 §R4): this loop used to call `llm.generate` once per
    query, which leaves an H200 almost idle -- D002 measured batching at 3.7-6.9x
    on the same hardware. Every query inside one `prompt_conf` shares that
    config's sampling hyper-parameters, so they can be decoded as one batch
    without changing any of them. Batching across *configs* would not be safe;
    batching within one is.

    Correctness relies on `LLM_huggingface`'s tokenizer being constructed with
    `padding_side='left'` (`llm.py:30`): the output slice in `generate` uses the
    padded input width, which is only the true prompt boundary under left
    padding. Verified empirically in `experiments/d006_s1_verify.py` (greedy
    batched output must be character-identical to unbatched).

    `max_new_tokens=None` defers to `llm.py`'s module default; D006 passes the
    C7-decided 200-token ceiling explicitly so the corpus's token budget lives
    at the call site and in the manifest rather than in a module global.
    """
    gen_kw = {} if max_new_tokens is None else {'max_new_tokens': max_new_tokens}

    entries = []
    for prompt_conf in tqdm.tqdm(prompt_confs):
        entry = {'dataset':pool, 'llm': llm.llm_name, 'traces': [], 'prompt_conf': prompt_conf.to_dict()}

        prompts, sample_params = [], None
        for query in queries:
            prompt, sample_params = prompt_conf(query, llm)
            prompts.append(prompt)

        outs = []
        for i in range(0, len(prompts), batch_size):
            outs.extend(llm.generate(prompts[i:i+batch_size], sample_params, **gen_kw))

        assert len(outs) == len(queries), \
            f"batched generation returned {len(outs)} outputs for {len(queries)} queries"
        entry['traces'] = list(zip(queries, outs))
        entries.append(entry)
    return entries

class DatasetMaker:

    def __init__(self, pc, llms, queries, num_prompt_conf_train, num_prompt_conf_test, output_path, encoding='utf8'):
        self.pc = pc
        self.llms = llms
        self.queries = queries
        self.num_prompt_conf_train = num_prompt_conf_train
        self.num_prompt_conf_test = num_prompt_conf_test
        self.output_path = output_path

        self.encoding = encoding
    
        self.train = []
        self.test = []

    def run_on_an_llm(self, llm_name, llm_type):

        train_prompt_conf = self.pc.sample(self.num_prompt_conf_train, pool=TRAIN)
        test_prompt_conf = self.pc.sample(self.num_prompt_conf_test, pool=TEST)
        
        print(f"Loading {llm_name}...")
        llm = load_llm(llm_name, llm_type)
        print(f"\tRunning on {llm_name} train...")
        _train = make_dataset_entries_for_new_llm(llm, self.queries, train_prompt_conf, pool=TRAIN)
        self.dump(_train)
        self.train += _train
        print(f"\tRunning on {llm_name} test...")
        _test = make_dataset_entries_for_new_llm(llm, self.queries, test_prompt_conf, pool=TEST)
        self.dump(_test)
        self.test += _test

    def __call__(self):
        for llm_name, llm_type in self.llms:
            self.run_on_an_llm(llm_name, llm_type)

    def dump(self, entries):
        with open(self.output_path, 'a', encoding=self.encoding) as f:
            for entry in entries:
                print(json.dumps(entry), file=f)
