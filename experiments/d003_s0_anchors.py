"""
D003 / P1 — S0: recover and verify the anchor query sets.

Blocking prerequisite for S2, which requires the two published baselines
VERBATIM. P1 F1 flagged that our PDF text extraction linearized Table F.1's two
columns and a semantic split gave 7/9=16 against the paper's stated 8+8.

This script takes the approved primary route: the arXiv HTML rendering, which
preserves table structure, so column assignment is READ rather than inferred.

Outcome (2026-09-02): HTML parse yields exactly 8 + 8, reconciling with the
paper. It also shows the semantic reconstruction would have been WRONG --
"Write a creative story involving quantum mechanics and a detective mystery."
sits in GPT4o-gen-opt, not random-opt as its Alpaca-like phrasing suggests.
That is the failure P1 S0's stop-and-verify posture existed to prevent.
"""
import re
import json
import html
import argparse
import urllib.request

HTML_URL = "https://arxiv.org/html/2407.15847v4"
CAPTION = r"[Bb]aseline query strateg"


def parse_table_f1(page: str):
    m = list(re.finditer(CAPTION, page))
    if not m:
        raise SystemExit("STOP: Table F.1 caption not found in HTML.")
    cap = m[-1].start()
    start, end = page.rfind("<table", 0, cap), page.find("</table>", cap - 20000)
    if start < 0 or end < 0:
        raise SystemExit("STOP: could not bound the Table F.1 <table> element.")
    tbl = page[start:page.find("</table>", start)]

    def cells(row):
        out = []
        for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S):
            t = html.unescape(re.sub(r"<[^>]+>", "", c))
            out.append(re.sub(r"\s+", " ", t).strip())
        return out

    rows = [cells(r) for r in re.findall(r"<tr.*?</tr>", tbl, re.S)]
    rows = [r for r in rows if len(r) == 2 and any(r)]
    if not rows:
        raise SystemExit("STOP: no 2-column rows parsed from Table F.1.")

    header, body = rows[0], rows[1:]
    if "gpt4o" not in header[0].lower() or "random" not in header[1].lower():
        raise SystemExit(f"STOP: unexpected header {header!r}")
    return [r[0] for r in body if r[0]], [r[1] for r in body if r[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./results/D003/anchors.json")
    ap.add_argument("--paper8", default="./confs/queries/default.json")
    args = ap.parse_args()

    page = urllib.request.urlopen(HTML_URL, timeout=60).read().decode("utf-8", "replace")
    gpt4o, rand = parse_table_f1(page)
    paper8 = json.load(open(args.paper8))

    # The paper states the greedy procedure converged to 8 queries. P1 S0:
    # stop and report rather than ship a silently-wrong anchor set.
    problems = []
    if len(gpt4o) != 8:
        problems.append(f"gpt4o-gen-opt has {len(gpt4o)}, expected 8")
    if len(rand) != 8:
        problems.append(f"random-opt has {len(rand)}, expected 8")
    if len(paper8) != 8:
        problems.append(f"paper default.json has {len(paper8)}, expected 8")
    if problems:
        raise SystemExit("STOP -- anchor split indefensible: " + "; ".join(problems))

    anchors = []
    for txt in paper8:
        anchors.append(dict(text=txt, provenance="anchor", anchor_set="paper8",
                            source="confs/queries/default.json",
                            confidence="exact"))
    for txt in gpt4o:
        anchors.append(dict(text=txt, provenance="anchor", anchor_set="gpt4o-gen-opt",
                            source=HTML_URL, confidence="exact"))
    for txt in rand:
        anchors.append(dict(text=txt, provenance="anchor", anchor_set="random-opt",
                            source=HTML_URL, confidence="exact"))

    out = dict(
        source="html", url=HTML_URL,
        note=("Column assignment READ from HTML table structure, not inferred. "
              "A semantic split would have mis-assigned at least one entry "
              "(the 'creative story / quantum mechanics' query is GPT4o-gen-opt, "
              "not random-opt)."),
        counts={"paper8": len(paper8), "gpt4o-gen-opt": len(gpt4o),
                "random-opt": len(rand), "total": len(anchors)},
        anchors=anchors,
    )
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out["counts"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
