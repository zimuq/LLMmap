import os
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from anthropic import Anthropic

max_new_tokens = 100
CACHE_DIR = os.environ.get('HF_MODEL_CACHE', None)

class LLM_huggingface:
    
    def __init__(
        self,
        llm_name,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        model_load_kargs={},
        tokenizer_only=False,
        tokenizer_load_kargs=None,
    ):
        # D006/S4: `model_load_kargs` was passed verbatim to BOTH the tokenizer
        # and the model, so any tokenizer-only argument also reached the model
        # constructor. Most models ignore an unknown kwarg; custom remote-code
        # classes do not -- InternLM2ForCausalLM raises
        # `unexpected keyword argument 'use_fast'`. `tokenizer_load_kargs` adds
        # arguments for the TOKENIZER ONLY. Default None preserves the previous
        # behaviour exactly, so no existing caller changes.

        api_key = os.environ.get('HUGGINGFACE_API_KEY', None)
        if api_key is None:
            raise Exception(f'Missing HuggingFace APIs key. Export "HUGGINGFACE_API_KEY" in the enverioment and try again')
            
        self.llm_name = llm_name
        self.model_class = model_class
        
        _tok_kargs = dict(model_load_kargs)
        if tokenizer_load_kargs:
            _tok_kargs.update(tokenizer_load_kargs)
        self.tokenizer = tokenizer_class.from_pretrained(llm_name, padding_side='left', token=api_key, legacy=False, **_tok_kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token            
        self.tokenizer.with_system_prompt = True
        
        self.is_hf = True
        # Computed once: make_prompt is called for every (query, config) pair, and
        # the probe renders a template each time it runs.
        self.supports_system_role = self._does_template_have_system(self.tokenizer)

        self.model = None
        if not tokenizer_only:
            self.model = model_class.from_pretrained(llm_name, token=api_key, **model_load_kargs)
            self.model.generation_config.pad_token_ids = self.tokenizer.pad_token_id

    _SYS_SENTINEL = "__cdqd_system_probe__"

    @classmethod
    def _does_template_have_system(cls, tokenizer):
        """Does this chat template actually accept AND render a system role?

        D006/S4: the original test was `"system" in chat_template` -- a substring
        match on the template source. That is wrong in both directions, and
        measurably so: on D006's 37-model universe it disagrees with reality for
        **10 of them**.

          * gemma's template contains the word "system" only inside
            `raise_exception('System role not supported')`, so the substring test
            says yes and the render then raises. All 5 gemma models crashed.
          * `Llama3-ChatQA-1.5-8B` likewise tests True but silently DROPS the
            system content -- so ~90% of its configs (WITH_SYSTEM_P=0.9) were
            generated with the system prompt missing entirely, while every other
            model received it. A silent data defect, not a crash.
          * `aya-23-8B`, `Llama-3-8B-Gradient-1048k`, `Meta-Llama-3-8B-Instruct`
            and `openchat_3.5` test False but DO support a system role, so their
            system prompt was prepended to the user turn instead.

        Rendering a sentinel and checking it survives tests the behaviour rather
        than the source. Both halves matter: a template that raises fails, and so
        does one that accepts the message and discards it.
        """
        try:
            out = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': cls._SYS_SENTINEL},
                 {'role': 'user', 'content': 'probe'}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return False
        return cls._SYS_SENTINEL in out

    def make_prompt(self, system, user):
        messages = []
        if system:
            if self.supports_system_role:
                messages.append( {'role':'system', 'content':system} )
            else:
                user = f'{system}\n\n{user}'
                
        messages.append( {'role':'user', 'content':user} )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text
            
    def generate(
        self,
        prompt,
        gen_kargs,
        skip_special_tokens=True,
        max_new_tokens=max_new_tokens,
    ):

        with torch.no_grad():
            in_toks = self.tokenizer(
                prompt,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                return_token_type_ids=False
            ).to(self.model.device)
            out_toks = self.model.generate(
                **in_toks,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kargs
            )
    
            gen_toks = [out_toks[i,in_toks.input_ids[i].shape[0]:] for i in range(len(out_toks))]
            gen_strs = self.tokenizer.batch_decode(gen_toks, skip_special_tokens=skip_special_tokens)

        return gen_strs


#####################################################################################################################

class LLM_OpenAI:
    def __init__(self, llm_name):
                
        api_key = os.environ.get('OPENAI_API_KEY', None)
        if api_key is None:
            raise Exception(f'Missing OpenAPI APIs key. Export "OPENAI_API_KEY" in the enverioment and try again')
        
        self.client = OpenAI(api_key=api_key)
        self.llm_name = llm_name
        
        self.is_hf = False


    def make_prompt(self, system, user):
        messages = []
        if system:
            messages += [{'role':'system', 'content':system}]
        messages += [{'role':'user', 'content':user}]
        return messages


    def _convert_gen_kargs(self, gen_kargs):
        if 'do_sample' in gen_kargs:
            do_sample = gen_kargs.pop('do_sample')
            if not do_sample:
                gen_kargs['temperature'] = 0

        return gen_kargs

        
    def generate(
        self,
        prompt,
        gen_kargs,
        max_new_tokens=max_new_tokens
    ):

        gen_kargs = self._convert_gen_kargs(gen_kargs)
        gen_kargs['max_tokens'] = max_new_tokens
        
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=prompt,
            **gen_kargs,
        )
        output = response.choices[0].message.content
        return [output]
##################################################################################################################### 


class LLM_Anthropic(LLM_OpenAI):
    def __init__(self, llm_name):

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise RuntimeError('Missing Anthropic API key. Export "ANTHROPIC_API_KEY" and try again.')
                
        self.client = client = Anthropic(api_key=api_key)
        self.llm_name = llm_name
        self.is_hf = False

    def make_prompt(self, system, user):
        messages = [{'role':'user', 'content':user}]
        return (system, messages)
    
    def generate(
        self,
        prompt,
        gen_kargs,
        max_new_tokens=max_new_tokens
    ):

        system, messages = prompt
        gen_kargs = self._convert_gen_kargs(gen_kargs)
        
        message = self.client.messages.create(
            max_tokens=max_new_tokens,
            system=system, 
            messages=messages,
            model=self.llm_name,
            **gen_kargs,
        )
        
        out = message.content[0].text
        
        return [out]

##################################################################################################################### 


def load_llm(llm_name, llm_type, cache_dir=CACHE_DIR, **kargs):

    if llm_type == 0:
        kargs['model_load_kargs'] = {'device_map':"auto", 'cache_dir':cache_dir, 'trust_remote_code':True}
        llm = LLM_huggingface(llm_name, **kargs)
    elif llm_type == 1:
        llm = LLM_OpenAI(llm_name)
    elif llm_type == 2:
        llm = LLM_Anthropic(llm_name)
    else:
        raise Exception()
    return llm