"""Controls: decoder-only training against FIXED encoders (no joint training).
 grid   : hand-designed 8x4 black/white block code (conventional, non-neural format)
 randenc: frozen randomly initialised Encoder (seed e); never trained.
For each fixed encoder, two independently seeded decoders are trained (same data, schedule); we cross-evaluate
decoder1/decoder2 on that encoder's codes. Tests whether cross-decoder failure needs joint training."""
import json, sys, numpy as np, torch
from common import *
torch.set_num_threads(3); S = make_splits(); test = S["test"]; train = torch.from_numpy(S["train"]).float()
def grid(bits):
    b = np.asarray(bits, np.float32).reshape(-1, 4, 8); im = np.kron(b, np.ones((16, 8), np.float32))
    return 0.1 + 0.8 * im   # 64x64
class Grid(torch.nn.Module):
    def forward(s, m): return torch.from_numpy(grid(m.numpy()))[:, None]
def make(kind):
    if kind == "grid": return Grid()
    torch.manual_seed(int(kind[7:])); return Encoder().eval()
def train_dec(enc, seed, steps=2000):
    torch.manual_seed(seed); d = Decoder(); ch = DiffChannel(); opt = torch.optim.Adam(d.parameters(), 1e-3)
    sc = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05); bce = torch.nn.BCEWithLogitsLoss(); rs = np.random.RandomState(seed)
    for st in range(steps):
        m = train[rs.randint(0, len(train), 128)]
        with torch.no_grad(): x = enc(m)
        s = torch.rand(128) * min(1, .15 + .85 * st / (.4 * steps))
        l = bce(d(ch(x, s)), m); opt.zero_grad(); l.backward(); opt.step(); sc.step()
    return d.eval()
R = {}
for kind in ["grid", "randenc1", "randenc2"]:
    enc = make(kind); x = (enc(torch.from_numpy(test).float()).detach()[:, 0].numpy() * 255).round() / 255
    ds = [train_dec(enc, s) for s in (1, 2)]
    R[kind] = {}
    for lvl in ["digital", "L2"]:
        xl = np_channel(x, LEVELS[lvl], 11)
        R[kind][lvl] = {f"dec{i+1}": metrics(decode(d, xl), test) for i, d in enumerate(ds)}
    print(kind, {l: {k: round(v["ber"], 4) for k, v in R[kind][l].items()} for l in R[kind]}, flush=True)
json.dump(R, open(f"{RES}/ablation_fixed_encoder.json", "w"), indent=1)
