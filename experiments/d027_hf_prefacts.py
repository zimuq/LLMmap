import json, os, sys, re
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, EntryNotFoundError
import jinja2, jinja2.meta
tok = open('/work/11280/zimuq1/vista/.hf_token').read().strip()
api = HfApi(token=tok)
env = jinja2.Environment()
out = {}
for m in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    r = dict()
    try:
        info = api.model_info(m, files_metadata=True)
        r['sha'] = info.sha; r['gated'] = info.gated
        files = {s.rfilename: s.size for s in info.siblings}
        r['weights_gb'] = round(sum(v or 0 for k, v in files.items() if k.endswith('.safetensors'))/1e9, 1)
        r['has_bin_only'] = not any(k.endswith('.safetensors') for k in files)
        r['has_jinja'] = 'chat_template.jinja' in files
        r['remote_code_py'] = sorted(k for k in files if k.endswith('.py'))
        r['has_tekken'] = 'tekken.json' in files
        r['has_tokenizer_config'] = 'tokenizer_config.json' in files
        r['has_processor'] = any(k in files for k in ('processor_config.json','preprocessor_config.json'))
    except Exception as e:
        r['info_error'] = repr(e)[:200]; out[m] = r; print(m, r); continue
    try:
        c = json.load(open(hf_hub_download(m, 'config.json', token=tok)))
        r['access'] = True
        r['arch'] = c.get('architectures'); r['model_type'] = c.get('model_type'); r['tf_ver'] = c.get('transformers_version')
        r['auto_map'] = bool(c.get('auto_map'))
        tc = c.get('text_config') or {}
        r['max_pos'] = c.get('max_position_embeddings') or tc.get('max_position_embeddings')
    except GatedRepoError as e:
        r['access'] = False; r['access_err'] = 'gated, no access'
    except Exception as e:
        r['access'] = False; r['access_err'] = repr(e)[:200]
    tmpl = None
    if r.get('access'):
        try:
            if r.get('has_jinja'):
                tmpl = open(hf_hub_download(m, 'chat_template.jinja', token=tok)).read()
            elif r.get('has_tokenizer_config'):
                t = json.load(open(hf_hub_download(m, 'tokenizer_config.json', token=tok))).get('chat_template')
                if isinstance(t, list):
                    t = next((x['template'] for x in t if x.get('name') == 'default'), t[0]['template'])
                tmpl = t
            if tmpl is None and 'chat_template.json' in files:
                tmpl = json.load(open(hf_hub_download(m, 'chat_template.json', token=tok))).get('chat_template')
        except Exception as e:
            r['tmpl_err'] = repr(e)[:200]
    r['has_template'] = tmpl is not None
    if tmpl:
        try:
            vars_ = jinja2.meta.find_undeclared_variables(env.parse(tmpl))
        except Exception as e:
            vars_ = {'PARSE_ERR:' + repr(e)[:80]}
        r['tmpl_vars'] = sorted(v for v in vars_ if v not in ('messages','add_generation_prompt','bos_token','eos_token','raise_exception','tools','pad_token','unk_token'))
        r['strftime'] = 'strftime_now' in tmpl
        r['think_markers'] = sorted(set(re.findall(r'<\|?/?(?:think|thinking|reasoning|channel)[^>]*>', tmpl)))[:6]
        r['mentions_system'] = 'system' in tmpl
        os.makedirs('tmpl', exist_ok=True); open('tmpl/' + m.replace('/', '__') + '.jinja', 'w').write(tmpl)
    out[m] = r
    print(m, {k: v for k, v in r.items() if k not in ('remote_code_py',)}, flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
