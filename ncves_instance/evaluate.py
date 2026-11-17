"""Experiments A/B/C/E: decode held-out codes of the ORIGINAL encoder with every trained decoder,
under digital + simulated physical channels. Also cross matrix and costs."""
import json, os, time, numpy as np, torch
from common import *
torch.set_num_threads(2)
test = make_splits()["test"]
names = sorted(n for n in os.listdir(RUNS) if os.path.exists(f"{RUNS}/{n}/config.json"))
inst = {n: load_instance(n) for n in names}
conds = {k: (s, None) for k, s in LEVELS.items()}
conds |= {f"{f}@1": (1.0, f) for f in FACTORS}
codes = {n: render(inst[n][0], test) for n in names}
deg = {c: np_channel(codes["orig"], s, 11, f) for c, (s, f) in conds.items()}
main = {}
for n in names:
    main[n] = {c: metrics(decode(inst[n][1], deg[c]), test) for c in conds}
# cross matrix (source encoder x decoder), BER only, digital and L2
cross = {}
for cond in ["digital", "L2"]:
    cross[cond] = {}
    for src in names:
        x = np_channel(codes[src], conds[cond][0], 11)
        cross[cond][src] = {d: float((decode(inst[d][1], x) != test).mean()) for d in names}
# costs
e, d, _ = inst["orig"]; x = torch.from_numpy(codes["orig"][:1, None]); m = torch.from_numpy(test[:1]).float()
def lat(f, n=200):
    f(); t = time.time(); [f() for _ in range(n)]; return (time.time() - t) / n * 1000
with torch.no_grad():
    cost = dict(enc_params=nparams(e), dec_params=nparams(d), enc_bytes=os.path.getsize(f"{RUNS}/orig/enc.pt"),
                dec_bytes=os.path.getsize(f"{RUNS}/orig/dec.pt"), payload_bits=NBITS, info_bits=NINFO,
                bits_per_pixel=NBITS / IMG ** 2, decode_ms_per_code_cpu1thread=lat(lambda: d(x)),
                encode_ms_per_code_cpu1thread=lat(lambda: e(m)),
                train_seconds_orig=json.load(open(f"{RUNS}/orig/config.json"))["train_seconds"])
json.dump(dict(main=main, cross=cross, cost=cost, conds=list(conds), n_test=len(test)), open(f"{RES}/evaluate.json", "w"), indent=1)
for n in names: print(f"{n:22s}", " ".join(f"{c}:{main[n][c]['ber']:.3f}" for c in ["digital", "L2", "L3"]))
print(cost)
