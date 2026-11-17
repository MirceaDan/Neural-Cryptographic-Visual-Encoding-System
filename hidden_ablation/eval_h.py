"""Cross-decoding matrices for Group B (H0-4), Group A (ncves orig,B_r1-5: reused from ../ncves_instance/results/evaluate.json),
Group C (K0_*, K1_*), + wrong-key tests + paired statistics + NCVES-vs-HiDDeN-style comparison of the gap."""
import json, os, numpy as np, torch, time
from scipy import stats
from lib_h import *
torch.set_num_threads(2); S = make_splits(); CV = covers(); test = S["test"]; ctest = CV["test"]
names = sorted(n for n in os.listdir(RUNS_H) if os.path.exists(f"{RUNS_H}/{n}/config.json")); inst = {n: load(n) for n in names}
R = {"models": names}
def matrix(group, cond):
    codes = {n: render_h(inst[n][0], inst[n][2], test, ctest) for n in group}; M = {}
    for i in group:
        x = np_channel(codes[i], LEVELS[cond], 11); M[i] = {}
        for j in group:
            m = metrics(decode(inst[j][1], x), test); M[i][j] = dict(ber=m["ber"], exact=m["exact"], crc=m["crc_accept"])
    return M
def gap(M):
    g = list(M); d = np.array([M[i][i]["ber"] for i in g]); o = np.array([np.mean([M[i][j]["ber"] for j in g if j != i]) for i in g])
    hx = M[g[0]][g[0]]["exact"] is not None
    de = np.array([M[i][i]["exact"] for i in g]) if hx else np.array([0.]); oe = np.array([np.mean([M[i][j]["exact"] for j in g if j != i]) for i in g]) if hx else np.array([0.])
    diff = o - d; ci = stats.t.interval(.95, len(g) - 1, diff.mean(), stats.sem(diff)) if len(g) > 2 else [float("nan")] * 2
    return dict(n_pairs=len(g), diag_ber=float(d.mean()), diag_ber_sd=float(d.std(ddof=1)), off_ber=float(o.mean()), off_ber_sd=float(o.std(ddof=1)),
                gap=float(diff.mean()), gap_ci95=[float(c) for c in ci], diag_exact=float(de.mean()), off_exact=float(oe.mean()),
                off_ber_min=float(min(M[i][j]["ber"] for i in g for j in g if i != j)), off_exact_max=float(max(M[i][j]["exact"] for i in g for j in g if i != j)) if hx else None,
                off_crc_mean=float(np.mean([M[i][j]["crc"] for i in g for j in g if i != j])) if hx else None)
groups = {"B_hidden": [n for n in names if n.startswith("H") and not n.startswith("HL")], "B2_hidden_lowimg": [n for n in names if n.startswith("HL")], "C_key0": [n for n in names if n.startswith("K0")], "C_key1": [n for n in names if n.startswith("K1")]}
R["matrices"] = {}; R["gap"] = {}
for g, mem in groups.items():
    if len(mem) < 2: continue
    for cond in ["digital", "L2"]:
        M = matrix(mem, cond); R["matrices"][f"{g}|{cond}"] = M; R["gap"][f"{g}|{cond}"] = gap(M)
# Group A (reused): orig + B_r1..5 digital/L2 cross matrices from first study
E = json.load(open(os.path.join(H, "..", "ncves_instance", "results", "evaluate.json"))); A = ["orig", "B_r1", "B_r2", "B_r3", "B_r4", "B_r5"]
for cond in ["digital", "L2"]:
    MA = {i: {j: dict(ber=E["cross"][cond][i][j], exact=None, crc=None) for j in A} for i in A}; R["matrices"][f"A_ncves|{cond}"] = MA
    R["gap"][f"A_ncves|{cond}"] = {k: v for k, v in gap(MA).items() if "exact" not in k and "crc" not in k}  # BER only; exact not stored in evaluate.json cross
# Group C: wrong-key behaviour with own decoder, and cross-key transfer
Rk = {}
for n in names:
    if n.startswith("K"):
        e, d, cfg = inst[n]; k = secret_key(cfg["init_seed"]); rk = torch.from_numpy(np.random.RandomState(5).randint(0, 2, (len(test), KEYBITS))).float()
        good = render_h(e, cfg, test, key=k)
        with torch.no_grad():
            wk = (e(torch.from_numpy(test).float(), rk)[:, 0].numpy() * 255).round() / 255
            hk = k.clone(); hk[0, 0] = 1 - hk[0, 0]; w1 = (e(torch.from_numpy(test).float(), hk.expand(len(test), -1))[:, 0].numpy() * 255).round() / 255
        Rk[n] = {c: dict(correct_key=metrics(decode(d, np_channel(good, LEVELS[c], 11)), test), random_wrong_key=metrics(decode(d, np_channel(wk.astype(np.float32), LEVELS[c], 11)), test),
                         one_bit_flipped_key=metrics(decode(d, np_channel(w1.astype(np.float32), LEVELS[c], 11)), test)) for c in ["digital", "L2"]}
R["key_tests"] = Rk
# latency
e, d, cfg = inst[names[0]]; x = torch.from_numpy(render_h(e, cfg, test[:1], ctest[:1])[None])
with torch.no_grad():
    d(x); t = time.time(); [d(x) for _ in range(200)]; R["decode_ms"] = (time.time() - t) / 200 * 1000
# H legit under single factors + cover fidelity
for n in names:
    if n.startswith("H"):
        e, d, cfg = inst[n]; code = render_h(e, cfg, test, ctest)
        R.setdefault("hidden_legit", {})[n] = dict(psnr_vs_cover=float(10 * np.log10(1 / ((code - ctest) ** 2).mean())), **{c: metrics(decode(d, np_channel(code, LEVELS[c], 11)), test) for c in LEVELS})
json.dump(R, open(f"{RES_H}/eval_h.json", "w"), indent=1)
for k, v in R["gap"].items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
