"""Builds results/tables.md from evaluate.json, attacks/*.json, attack_unlabeled.json, analysis.json."""
import json, glob, os, numpy as np
from common import RES, tci, wilson

E = json.load(open(f"{RES}/evaluate.json")); M = E["main"]; out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
f = lambda v, d=4: f"{v:.{d}f}"
cell = lambda r: f"{f(r['ber'])} [{f(r['ber_ci'][0])},{f(r['ber_ci'][1])}] / {f(r['exact'],3)} / {f(r['crc_accept'],3)}"
CON = ["digital", "L2", "L3"]
P("Cell format: BER [95% bootstrap CI over 2000 test messages] / exact-message rate / CRC-accept rate.")
P("For the original decoder CRC-accept = 1-FRR; for any unauthorised decoder CRC-accept = FAR(CRC). Chance: BER 0.5, exact 2^-32, CRC-accept 2^-8=0.0039.\n")
P("## Table 1. Decoders applied to held-out codes from the ORIGINAL encoder\n")
P("| scenario | decoder | " + " | ".join(CON) + " |"); P("|---|---|" + "---|" * len(CON))
P("| A legitimate | orig | " + " | ".join(cell(M["orig"][c]) for c in CON) + " |")
for n in sorted(M):
    if n.startswith("B_"): P(f"| B independent retrain | {n} | " + " | ".join(cell(M[n][c]) for c in CON) + " |")
Bn = [n for n in M if n.startswith("B_")]
agg = lambda names, c, k: tci([M[n][c][k] for n in names])
P("| **B aggregate (n=%d)** | mean, 95%% t-CI across instances | " % len(Bn) + " | ".join(
    f"BER {f(agg(Bn,c,'ber')[0])} [{f(agg(Bn,c,'ber')[1][0])},{f(agg(Bn,c,'ber')[1][1])}] / exact max {f(max(M[n][c]['exact'] for n in Bn),4)} / CRC-acc mean {f(agg(Bn,c,'crc_accept')[0],4)}" for c in CON) + " |")
for n in sorted(M):
    if n.startswith("C_"): P(f"| C variation | {n} | " + " | ".join(cell(M[n][c]) for c in CON) + " |")
A = {}
for p in glob.glob(f"{RES}/attacks/*.json"):
    j = json.load(open(p)); A[j["name"]] = j["res"]
groups = {}
for k in A:
    g = k.split("_s")[0] + (("_" + k.split("_", 3)[-1]) if "_from" in k or "_w" in k.split("_s")[-1] else "")
    groups.setdefault(k.rsplit("_s", 1)[0] + ("" if "_from" not in k and "_w" not in k else "|" + k.split("_s", 1)[1][2:]), []).append(k)
for g in sorted(groups, key=lambda g: (("|" in g), int(g.split("_n")[1].split("|")[0]))):
    ks = groups[g]; lab = {"": "labeled substitute"}.get("", "")
    cells = []
    for c in CON:
        b = tci([A[k][c]["ber"] for k in ks]); ex = max(A[k][c]["exact"] for k in ks); cr = np.mean([A[k][c]["crc_accept"] for k in ks])
        cells.append(f"BER {f(b[0])} [{f(b[1][0])},{f(b[1][1])}] / exact max {f(ex,4)} / CRC-acc {f(cr,4)}")
    P(f"| D attacker, labeled pairs | {g} (n_seeds={len(ks)}) | " + " | ".join(cells) + " |")
U = json.load(open(f"{RES}/attack_unlabeled.json")) if os.path.exists(f"{RES}/attack_unlabeled.json") else {}
for rep in U:
    for K in [0, 32, 100, 1000, 2000]:
        r = U[rep]; P(f"| D unlabeled codes + K labels | {rep}, K={K} | " + cell(r[f"K{K}_digital"]) + " | " + cell(r[f"K{K}_L2"]) + " | n/a |")

P("\n## Table 2. Single-factor physical-channel simulation (s=1), BER: original vs B mean (n=%d)\n" % len(Bn))
fc = [c for c in E["conds"] if c.endswith("@1")]
P("| decoder | " + " | ".join(fc) + " |"); P("|---|" + "---|" * len(fc))
P("| orig | " + " | ".join(f(M["orig"][c]["ber"]) for c in fc) + " |")
P("| B mean | " + " | ".join(f(np.mean([M[n][c]["ber"] for n in Bn])) for c in fc) + " |")
P("| orig exact | " + " | ".join(f(M["orig"][c]["exact"], 3) for c in fc) + " |")

P("\n## Table 3. Cross matrix BER, digital (rows: encoder that made the codes; cols: decoder)\n")
X = E["cross"]["digital"]; cols = list(X)
P("| enc \\ dec | " + " | ".join(cols) + " |"); P("|---|" + "---|" * len(cols))
for s in cols: P(f"| {s} | " + " | ".join(f(X[s][d], 3) for d in cols) + " |")
diag = np.mean([X[s][s] for s in cols]); off = np.mean([X[s][d] for s in cols for d in cols if s != d])
P(f"\nMean diagonal BER {f(diag)}; mean off-diagonal BER {f(off)} (digital).")
P("\n## Cost\n"); P("```\n" + json.dumps(E["cost"], indent=1) + "\n```")
open(f"{RES}/tables.md", "w").write("\n".join(out)); print("\n".join(out))
