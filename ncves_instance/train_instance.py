"""Train one encoder/decoder instance jointly through the differentiable channel.
Separate seeds: init (weights), data (batch order), noise (channel randomness)."""
import argparse, json, os, time, sys
import numpy as np, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--init_seed", type=int, default=0)
ap.add_argument("--data_seed", type=int, default=0)
ap.add_argument("--noise_seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--width", type=int, default=1)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
out = os.path.join(RUNS, a.name)
if os.path.exists(out):
    sys.exit(f"refusing to overwrite existing {out}")
os.makedirs(out)
train = torch.from_numpy(make_splits()["train"]).float()
val = make_splits()["val"]

torch.manual_seed(a.init_seed)
enc, dec = Encoder(a.width), Decoder(a.width)
torch.manual_seed(a.noise_seed + 10_000)  # channel randomness stream, independent of init
ch = DiffChannel()
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
rs = np.random.RandomState(a.data_seed); perm = rs.permutation(len(train)); pos = 0
logf = open(os.path.join(out, "log.jsonl"), "w"); t0 = time.time(); bce = torch.nn.BCEWithLogitsLoss()
for step in range(a.steps):
    if pos + a.batch > len(perm): perm = rs.permutation(len(train)); pos = 0
    m = train[perm[pos:pos + a.batch]]; pos += a.batch
    cap = min(1.0, 0.15 + 0.85 * step / (0.4 * a.steps))  # severity curriculum
    s = torch.rand(a.batch) * cap * (torch.rand(a.batch) > 0.1)
    logits = dec(ch(enc(m), s))
    loss = bce(logits, m)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 100 == 0 or step == a.steps - 1:
        ber = ((logits > 0).float() != m).float().mean().item()
        logf.write(json.dumps(dict(step=step, loss=loss.item(), train_ber=ber, cap=cap, t=time.time() - t0)) + "\n"); logf.flush()
torch.save(enc.state_dict(), os.path.join(out, "enc.pt")); torch.save(dec.state_dict(), os.path.join(out, "dec.pt"))
enc.eval(); dec.eval()
r = {k: metrics(decode(dec, np_channel(render(enc, val), s, 7)), val) for k, s in LEVELS.items()}
cfg = vars(a) | dict(train_seconds=time.time() - t0, val=r, enc_params=nparams(enc), dec_params=nparams(dec),
                     sha_enc=sha256(os.path.join(out, "enc.pt")), sha_dec=sha256(os.path.join(out, "dec.pt")))
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
print(a.name, {k: (round(v["ber"], 4), round(v["exact"], 3)) for k, v in r.items()}, f"{time.time()-t0:.0f}s")
