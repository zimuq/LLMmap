"""D033 / S3 diagnostic (post-analysis): unrounded 2.5 / 97.5 percentiles of the two v1 readout-(2) secondary
contrasts whose 4-decimal interval endpoint is 0.0000 (R §1, §6 item 6). Re-reads the stored S_test terms; no label
is changed by this script.
Usage:  PYTHONPATH=.:experiments python experiments/d033_edge_ci.py
"""
import json, numpy as np, d033_analysis as A
ch = json.load(open(f"{A.RES}/chains.json")); vc = json.load(open(f"{A.RES}/variant_choice.json"))
V = A.View("v1", ch, vc)
for kind, a, b in (("family", "OURS", "JG"), ("model", "FULL", "OURS")):
    da, db = V.unit("nc", kind, a).stats(V.B)["all"][0], V.unit("nc", kind, b).stats(V.B)["all"][0]
    x = da - db
    print(kind, a, b, "p2.5 %.8f p97.5 %.8f  share==0 %.4f" % (np.percentile(x, 2.5), np.percentile(x, 97.5), np.mean(x == 0)), flush=True)
