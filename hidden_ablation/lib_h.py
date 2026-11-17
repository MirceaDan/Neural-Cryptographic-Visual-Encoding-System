"""Ablation vs HiDDeN-style joint training. Reuses ../ncves_instance/common.py (splits, Decoder, channels, metrics).
Kinds: hidden = HiDDeN-style cover-image encoder (ADAPTED, not faithful); key0 / key1 = repo-style key-conditioned
NCVES encoder with fixed secret key, without / with bounded key-separation loss."""
import os, sys, json, numpy as np, torch, torch.nn as nn, cv2
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "ncves_instance"))
import common as C
from common import *
RUNS_H, RES_H = os.path.join(H, "runs"), os.path.join(H, "results")
KEYBITS = 64

def covers(path=os.path.join(H, "covers.npz")):
    """Deterministic synthetic grayscale 'natural-like' covers (multi-scale smooth noise); no external dataset available offline.
    train 20000 / attacker 600 / test 2000, disjoint by construction (distinct RNG streams)."""
    if os.path.exists(path): return dict(np.load(path))
    def gen(seed, n):
        r = np.random.RandomState(seed); out = np.zeros((n, IMG, IMG), np.float32)
        for i in range(n):
            x = np.zeros((IMG, IMG), np.float32)
            for sc, a in [(3, 1.0), (6, .6), (12, .4), (24, .2)]:
                x += a * cv2.resize(r.randn(sc, sc).astype(np.float32), (IMG, IMG), interpolation=cv2.INTER_CUBIC)
            x = (x - x.min()) / (x.max() - x.min() + 1e-6); out[i] = 0.15 + 0.7 * x
        return out
    d = dict(train=gen(1, 20000), attacker=gen(2, 600), test=gen(3, 2000)); np.savez(path, **d); return d

class HEnc(nn.Module):  # HiDDeN-style: conv blocks on cover, message tiled spatially & concatenated, final conv -> image
    def __init__(s, w=1):
        super().__init__(); c = 16 * w
        def blk(i, o): return [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU()]
        s.feat = nn.Sequential(*blk(1, c), *blk(c, c)); s.mix = nn.Sequential(*blk(c + NBITS + 1, c), nn.Conv2d(c, 1, 1))
    def forward(s, m, cover):
        f = s.feat(cover); t = (m * 2 - 1)[:, :, None, None].expand(-1, -1, IMG, IMG)
        return torch.sigmoid(s.mix(torch.cat([f, t, cover], 1)))

class KEnc(Encoder):  # repo-style: message ++ key -> image
    def __init__(s, w=1):
        super().__init__(w); s.fc[0] = nn.Linear(NBITS + KEYBITS, 512)
    def forward(s, m, k): return super().forward(torch.cat([m, k], 1) if False else m) if False else torch.sigmoid(s.up(s.fc(torch.cat([m * 2 - 1, k * 2 - 1], 1)).view(-1, s.c, 8, 8)))

def secret_key(seed):
    return torch.from_numpy(np.random.RandomState(777 + seed).randint(0, 2, (1, KEYBITS))).float()

def load(name):
    d = os.path.join(RUNS_H, name); cfg = json.load(open(f"{d}/config.json")); kind = cfg["kind"]
    e = HEnc() if kind == "hidden" else KEnc(); dec = Decoder()
    e.load_state_dict(torch.load(f"{d}/enc.pt")); dec.load_state_dict(torch.load(f"{d}/dec.pt")); e.eval(); dec.eval()
    return e, dec, cfg

@torch.no_grad()
def render_h(e, cfg, bits, cov=None, key=None, bs=250):
    out = []
    for i in range(0, len(bits), bs):
        m = torch.from_numpy(bits[i:i + bs]).float()
        if cfg["kind"] == "hidden": x = e(m, torch.from_numpy(cov[i:i + bs, None]))
        else: x = e(m, (key if key is not None else secret_key(cfg["init_seed"])).expand(len(m), -1))
        out.append(x)
    return (torch.cat(out)[:, 0].numpy() * 255).round().astype(np.float32) / 255
