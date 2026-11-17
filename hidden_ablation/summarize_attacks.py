"""Aggregate substitute-decoder attacks (mean/sd over attacker seeds) for NCVES (reused) and ablation targets -> results/attack_summary.md"""
import json, glob, os, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); rows = collections.defaultdict(list)
for f in glob.glob(f"{H}/results/attacks/*.json"):
    d = json.load(open(f)); rows[(d["args"]["target"], d["args"]["n"])].append(d["res"])
for f in glob.glob(f"{H}/../ncves_instance/results/attacks/sup_n*_s*.json"):
    d = json.load(open(f)); a = d["args"]
    if a.get("init_from") is None and a.get("width", 1) == 1 and a["n"] in (25, 100, 500): rows[("NCVES orig", a["n"])].append(d["res"])
out = ["| target | N labeled | seeds | clean BER (mean±sd) | clean exact | L2 BER | L2 exact |", "|---|---|---|---|---|---|---|"]
for (t, n), rs in sorted(rows.items()):
    g = lambda c, k: np.array([r[c][k] for r in rs])
    out.append(f"| {t} | {n} | {len(rs)} | {g('digital','ber').mean():.4f}±{g('digital','ber').std(ddof=1) if len(rs)>1 else 0:.4f} | {g('digital','exact').mean():.3f} | {g('L2','ber').mean():.4f}±{g('L2','ber').std(ddof=1) if len(rs)>1 else 0:.4f} | {g('L2','exact').mean():.3f} |")
open(f"{H}/results/attack_summary.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
