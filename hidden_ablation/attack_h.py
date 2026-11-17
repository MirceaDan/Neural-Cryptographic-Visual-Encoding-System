"""Substitute-decoder attack, IDENTICAL protocol to ../ncves_instance/attack_train.py (fresh Decoder, width 1, 3000 steps,
Adam 1e-3 OneCycle, training channel, labeled pairs from the disjoint attacker pool, evaluated on held-out test)."""
import argparse, json, os, sys, numpy as np, torch
from lib_h import *
ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args(); torch.set_num_threads(a.threads); out = f"{RES_H}/attacks/{a.target}_n{a.n}_s{a.seed}.json"
if os.path.exists(out): sys.exit("exists")
S = make_splits(); CV = covers(); enc, _, cfg = load(a.target)
pool = S["attacker"][:a.n]; codes = torch.from_numpy(render_h(enc, cfg, pool, CV["attacker"][:a.n]))[:, None]; lab = torch.from_numpy(pool).float()
torch.manual_seed(1000 + a.seed); dec = Decoder(); ch = DiffChannel(); steps = a.steps
opt = torch.optim.Adam(dec.parameters(), 1e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=.05)
bce = torch.nn.BCEWithLogitsLoss(); rs = np.random.RandomState(a.seed)
for st in range(steps):
    idx = torch.from_numpy(rs.randint(0, a.n, 128)); s = torch.rand(128) * min(1., .15 + .85 * st / (.4 * steps)) * (torch.rand(128) > .1)
    l = bce(dec(ch(codes[idx], s)), lab[idx]); opt.zero_grad(); l.backward(); opt.step(); sched.step()
dec.eval(); test = S["test"]; tc = render_h(enc, cfg, test, CV["test"])
res = {c: metrics(decode(dec, np_channel(tc, s, 11)), test) for c, s in LEVELS.items()}
json.dump(dict(args=vars(a), res=res), open(out, "w"), indent=1); print(a.target, a.n, a.seed, {c: round(res[c]["ber"], 4) for c in LEVELS}, flush=True)
