"""D006/S6b — does L2-normalising the I5 embedding change Sep(.,.)?

PAPER_DEVIATIONS item 3 (corrected 2026-09-06) establishes that I5 is the
paper's own stage-1 embedding. The paper's released implementation
(LLMmap/embedding_model.py:15-26) mean-pools and does NOT normalise; S6
normalised. Normalisation discards magnitude, which a distance-based
statistic can be sensitive to -- and A3 was decided partly on the C7 pilot's
saturation result, computed on normalised vectors. Measure, don't assume.
"""
import json, glob, itertools, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

I5="intfloat/multilingual-e5-large-instruct"; N=600
random.seed(0)
models=[m for m in json.load(open("data/corpus_v1/corpus_manifest.json"))["models"]
        if m["status"]=="VALIDATED"][:6]
tok=AutoTokenizer.from_pretrained(I5)
mdl=AutoModel.from_pretrained(I5, torch_dtype=torch.float16).cuda().eval()

def embed(texts, norm):
    out=[]
    for i in range(0,len(texts),64):
        b=tok(texts[i:i+64],padding=True,truncation=True,max_length=512,
              add_special_tokens=True,return_tensors="pt").to("cuda")
        with torch.no_grad():
            h=mdl(**b).last_hidden_state
            m=b["attention_mask"].unsqueeze(-1).float()
            e=(h*m).sum(1)/m.sum(1).clamp(min=1e-9)
            if norm: e=torch.nn.functional.normalize(e,dim=-1)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)

def auc(X,Y):
    Z=np.vstack([X,Y]); y=np.r_[np.zeros(len(X)),np.ones(len(Y))]
    return float(np.mean([roc_auc_score(y[te],LogisticRegression(max_iter=2000)
        .fit(Z[tr],y[tr]).decision_function(Z[te]))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Z,y)]))
def energy(X,Y): return float(2*cdist(X,Y).mean()-cdist(X,X).mean()-cdist(Y,Y).mean())

texts={}
for s in models:
    rows=[json.loads(l) for l in open(s["shard_path"])]
    t=[a for r in rows for _,a in r["traces"]]
    texts[s["model"]]=random.sample(t,N)
print(f"{len(models)} models x {N} responses", flush=True)

res={}
for norm in (True, False):
    E={m:embed(t,norm) for m,t in texts.items()}
    mags=[float(np.linalg.norm(v,axis=1).mean()) for v in E.values()]
    a=[];e=[]
    for x,y in itertools.combinations(models,2):
        A,B=E[x["model"]],E[y["model"]]; a.append(auc(A,B)); e.append(energy(A,B))
    res["normalised" if norm else "raw (paper)"]=dict(
        auc_mean=round(float(np.mean(a)),4), auc_min=round(float(np.min(a)),4),
        frac_at_ceiling=round(float(np.mean(np.array(a)>0.999)),3),
        energy_mean=round(float(np.mean(e)),5),
        mean_vector_norm=round(float(np.mean(mags)),3), aucs=[round(x,4) for x in a])
    print(f"{'normalised' if norm else 'raw (paper)':14s} "
          f"auc={res[list(res)[-1]]['auc_mean']:.4f} "
          f"ceil={res[list(res)[-1]]['frac_at_ceiling']:.0%} "
          f"energy={res[list(res)[-1]]['energy_mean']:.5f} "
          f"|v|={res[list(res)[-1]]['mean_vector_norm']:.2f}", flush=True)

r,n=res["raw (paper)"],res["normalised"]
d=np.array(r["aucs"])-np.array(n["aucs"])
res["comparison"]=dict(auc_mean_delta=round(float(d.mean()),5),
    pairs_where_raw_higher=int((d>0).sum()), n_pairs=len(d),
    max_abs_pair_delta=round(float(np.abs(d).max()),5))
print("\ncomparison:", res["comparison"])
json.dump(res, open("results/D006/s6b_norm_check.json","w"), indent=1)
