"""Experiment D (labeled): attacker trains a fresh decoder on (message, code) pairs from the ORIGINAL
encoder. Attacker knows architecture, training code, channel family. Pairs come from the attacker pool
(disjoint from the test set). --init_from starts from another instance's decoder (warm start)."""
import argparse, json, os, sys, numpy as np, torch
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init_from"); ap.add_argument("--width", type=int, default=1)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args(); torch.set_num_threads(a.threads)
name = f"sup_n{a.n}_s{a.seed}" + (f"_from{a.init_from}" if a.init_from else "") + (f"_w{a.width}" if a.width != 1 else "")
os.makedirs(f"{RES}/attacks", exist_ok=True); out = f"{RES}/attacks/{name}.json"
if os.path.exists(out): sys.exit("exists")
S = make_splits(); enc, _, _ = load_instance("orig")
pool = S["attacker"][:a.n]; codes = torch.from_numpy(render(enc, pool))[:, None]; lab = torch.from_numpy(pool).float()
torch.manual_seed(1000 + a.seed); dec = Decoder(a.width)
if a.init_from: dec.load_state_dict(load_instance(a.init_from)[1].state_dict())
ch = DiffChannel(); steps = a.steps; lr = 1e-3 if not a.init_from else 3e-4
opt = torch.optim.Adam(dec.parameters(), lr=lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
bce = torch.nn.BCEWithLogitsLoss(); rs = np.random.RandomState(a.seed)
for step in range(steps):
    idx = torch.from_numpy(rs.randint(0, a.n, 128)); cap = min(1.0, 0.15 + 0.85 * step / (0.4 * steps))
    s = torch.rand(128) * cap * (torch.rand(128) > 0.1)
    loss = bce(dec(ch(codes[idx], s)), lab[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
dec.eval(); test = S["test"]; tc = render(enc, test)
res = {c: metrics(decode(dec, np_channel(tc, s, 11)), test) for c, s in LEVELS.items()}
# train-set (memorisation) check
res["train_ber_digital"] = float((decode(dec, render(enc, pool[:500])) != pool[:500]).mean())
json.dump(dict(name=name, args=vars(a), res=res), open(out, "w"), indent=1)
print(name, {c: round(res[c]["ber"], 4) for c in LEVELS})
