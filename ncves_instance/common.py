"""Shared code: data splits, models, differentiable (torch) channel, realistic
(numpy/cv2) evaluation channel, metrics. Instance-only design: NO key input."""
import os, json, math, hashlib, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
RES = os.path.join(HERE, "results")
NINFO, NCRC, NBITS, IMG = 24, 8, 32, 64


# ---------------------------------------------------------------- data
def crc8(info_bits):
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 7
        reg = ((reg << 1) ^ 0x07) & 0xFF if reg & 0x80 else (reg << 1) & 0xFF
    return [(reg >> (7 - i)) & 1 for i in range(8)]


def make_splits(path=os.path.join(HERE, "data_splits.npz")):
    """Fixed, disjoint splits of unique 24-bit payloads (+CRC8 = 32 bits).
    train: defender training set; attacker: pool the attacker may label/query;
    test: held-out for ALL evaluation."""
    if os.path.exists(path):
        return dict(np.load(path))
    rng = np.random.RandomState(20240601)
    ids = rng.choice(2 ** NINFO, 44000, replace=False)
    def to_bits(v):
        info = [(int(x) >> (NINFO - 1 - i)) & 1 for x in v for i in range(NINFO)]
        info = np.array(info, np.uint8).reshape(len(v), NINFO)
        return np.concatenate([info, np.array([crc8(r) for r in info], np.uint8)], 1)
    d = dict(train=to_bits(ids[:20000]), attacker=to_bits(ids[20000:40000]),
             test=to_bits(ids[40000:42000]), val=to_bits(ids[42000:]))
    sets = [set(map(bytes, d[k])) for k in d]
    assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4)), "leak"
    np.savez(path, **d)
    return d


def crc_ok(bits):
    return np.array([crc8(r[:NINFO]) == list(r[NINFO:]) for r in bits.astype(int)])


# ---------------------------------------------------------------- models
class Encoder(nn.Module):
    def __init__(self, w=1):
        super().__init__()
        c = 32 * w
        self.fc = nn.Sequential(nn.Linear(NBITS, 512), nn.ReLU(), nn.Linear(512, c * 64), nn.ReLU())
        self.c = c
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c, 16 * w, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16 * w, 8 * w, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(8 * w, 4 * w, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(4 * w, 1, 3, padding=1))

    def forward(self, m):
        return torch.sigmoid(self.up(self.fc(m * 2 - 1).view(-1, self.c, 8, 8)))


class Decoder(nn.Module):
    def __init__(self, w=1):
        super().__init__()
        def blk(i, o): return [nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(4, o), nn.ReLU()]
        self.net = nn.Sequential(*blk(1, 16 * w), *blk(16 * w, 32 * w), *blk(32 * w, 64 * w),
                                 *blk(64 * w, 64 * w), nn.Flatten(),
                                 nn.Linear(64 * w * 16, 256), nn.ReLU(), nn.Linear(256, NBITS))

    def forward(self, x):
        return self.net((x - 0.5) * 2)


def nparams(m): return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- differentiable channel (training)
_Q = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
                  dtype=torch.float32)
_k = torch.arange(8).float()
_DCT = torch.cos((2 * _k[None, :] + 1) * _k[:, None] * math.pi / 16) * torch.where(_k[:, None] == 0, math.sqrt(1 / 8), math.sqrt(2 / 8))

# Max distortion magnitudes at severity s=1 (shared by torch + cv2 channels).
MAXP = dict(persp=0.06, blur=1.5, contrast=0.5, bright=0.15, gamma=0.3, grad=0.25,
            noise=0.08, jpeg=3.0, occl=0.10)


def _ste_round(x): return x + (torch.round(x) - x).detach()


def jpeg_proxy(x, scale):
    """Differentiable (straight-through) 8x8 DCT-quantisation. scale: (B,) multiplier of std table."""
    B, _, H, W = x.shape
    b = (x * 255 - 128).view(B, H // 8, 8, W // 8, 8).permute(0, 1, 3, 2, 4)
    D = _DCT.to(x.device)
    c = D @ b @ D.T
    q = _Q.to(x.device)[None, None, None] * scale.view(B, 1, 1, 1, 1)
    c = _ste_round(c / q) * q
    b = D.T @ c @ D
    return ((b.permute(0, 1, 3, 2, 4).reshape(B, 1, H, W) + 128) / 255)


def homography_grid(B, H, W, jit, dev, gen=None):
    src = torch.tensor([[-1., -1], [1, -1], [1, 1], [-1, 1]], device=dev)
    off = (torch.rand(B, 4, 2, device=dev) * 2 - 1) * (jit.view(B, 1, 1) * 2)
    dst = src[None] + off
    A = torch.zeros(B, 8, 8, device=dev); r = torch.zeros(B, 8, device=dev)
    for i in range(4):
        u, v = dst[:, i, 0], dst[:, i, 1]; x, y = src[i]
        A[:, 2 * i] = torch.stack([u, v, torch.ones_like(u), torch.zeros_like(u), torch.zeros_like(u), torch.zeros_like(u), -u * x, -v * x], 1)
        A[:, 2 * i + 1] = torch.stack([torch.zeros_like(u), torch.zeros_like(u), torch.zeros_like(u), u, v, torch.ones_like(u), -u * y, -v * y], 1)
        r[:, 2 * i], r[:, 2 * i + 1] = x, y
    h = torch.linalg.solve(A, r.unsqueeze(-1)).squeeze(-1)
    Hm = torch.cat([h, torch.ones(B, 1, device=dev)], 1).view(B, 3, 3)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    g = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, -1, 3)
    p = g @ Hm.transpose(1, 2)
    return (p[..., :2] / p[..., 2:3]).view(B, H, W, 2)


class DiffChannel(nn.Module):
    """Randomised differentiable camera/print channel. severity s in [0,1] per sample."""
    def forward(self, x, s):
        B, _, H, W = x.shape; dev = x.device; P = MAXP
        u = lambda: torch.rand(B, device=dev)
        # occlusion
        area = u() * P["occl"] * s * (torch.rand(B, device=dev) < 0.5)
        ar = torch.exp((u() * 2 - 1) * 0.7)
        w = (torch.sqrt(area * ar) * W).clamp(max=W); h = (torch.sqrt(area / ar) * H).clamp(max=H)
        cx, cy = u() * W, u() * H
        xs = torch.arange(W, device=dev).view(1, 1, W); ys = torch.arange(H, device=dev).view(1, H, 1)
        m = ((xs - cx.view(B, 1, 1)).abs() < w.view(B, 1, 1) / 2) & ((ys - cy.view(B, 1, 1)).abs() < h.view(B, 1, 1) / 2)
        x = torch.where(m.unsqueeze(1), (u() < 0.5).float().view(B, 1, 1, 1).expand_as(x), x)
        # perspective
        x = F.grid_sample(x, homography_grid(B, H, W, P["persp"] * s * u(), dev), padding_mode="border", align_corners=True)
        # blur
        sig = (0.05 + P["blur"] * s * u()).view(B, 1)
        d = torch.arange(-4, 5, device=dev).float().view(1, 9)
        k = torch.exp(-d ** 2 / (2 * sig ** 2)); k = k / k.sum(1, keepdim=True)
        xr = F.pad(x.view(1, B, H, W), (4, 4, 4, 4), mode="reflect")
        xr = F.conv2d(xr, k.view(B, 1, 1, 9), groups=B)
        x = F.conv2d(xr, k.view(B, 1, 9, 1), groups=B).view(B, 1, H, W)
        # photometric
        v = lambda: (u() * 2 - 1).view(B, 1, 1, 1)
        s4 = s.view(B, 1, 1, 1)
        gam = torch.exp(v() * P["gamma"] * s4)
        x = (x.clamp(1e-4, 1) ** gam - 0.5) * (1 - P["contrast"] * s4 * u().view(B, 1, 1, 1)) + 0.5 + v() * P["bright"] * s4
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
        ang = u() * 2 * math.pi
        x = x + P["grad"] * s4 * u().view(B, 1, 1, 1) * (torch.cos(ang).view(B, 1, 1) * xx + torch.sin(ang).view(B, 1, 1) * yy).unsqueeze(1) * 0.5
        # sensor noise (read + shot)
        x = x + (P["noise"] * s * u()).view(B, 1, 1, 1) * torch.randn_like(x) * (0.6 + 0.4 * x.clamp(0, 1).sqrt())
        # jpeg proxy
        x = jpeg_proxy(x.clamp(0, 1), 0.4 + P["jpeg"] * s * u())
        return x.clamp(0, 1)


# ---------------------------------------------------------------- realistic numpy/cv2 channel (evaluation)
FACTORS = ["persp", "blur", "illum", "noise", "jpeg", "occl", "print"]


def np_channel(imgs, s, seed, only=None):
    """imgs (N,H,W) float [0,1] -> degraded. Real cv2 warp/blur/JPEG. `only` = single factor.
    s=0 => identity. NOT the training implementation (independent code path)."""
    if s == 0 and only is None:
        return imgs.copy()
    P = MAXP; out = np.empty_like(imgs); N, H, W = imgs.shape
    on = lambda f: only is None or only == f
    for i in range(N):
        r = np.random.RandomState(seed * 1000003 + i); x = imgs[i].astype(np.float32).copy()
        if on("occl") and r.rand() < (0.5 if only is None else 1.0):
            a = r.rand() * P["occl"] * s if only is None else P["occl"] * s; ar = math.exp((r.rand() * 2 - 1) * 0.7)
            w, h = min(W, math.sqrt(a * ar) * W), min(H, math.sqrt(a / ar) * H); cx, cy = r.rand() * W, r.rand() * H
            x[int(max(0, cy - h / 2)):int(cy + h / 2), int(max(0, cx - w / 2)):int(cx + w / 2)] = float(r.rand() < 0.5)
        if on("persp"):
            j = P["persp"] * s * (r.rand() if only is None else 1) * W
            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]]); dst = src + (r.rand(4, 2) * 2 - 1).astype(np.float32) * j
            x = cv2.warpPerspective(x, cv2.getPerspectiveTransform(src, dst), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if on("blur"):
            sg = 0.05 + P["blur"] * s * (r.rand() if only is None else 1)
            if sg > 0.3: x = cv2.GaussianBlur(x, (0, 0), sg)
        if on("illum"):
            g = (r.rand() * 2 - 1) * P["gamma"] * s; x = np.clip(x, 1e-4, 1) ** math.exp(g)
            x = (x - 0.5) * (1 - P["contrast"] * s * r.rand()) + 0.5 + (r.rand() * 2 - 1) * P["bright"] * s
            yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij"); an = r.rand() * 2 * math.pi
            x = x + P["grad"] * s * r.rand() * 0.5 * (math.cos(an) * xx + math.sin(an) * yy)
        if on("noise"):
            sd = P["noise"] * s * (r.rand() if only is None else 1)
            x = x + sd * r.randn(H, W) * (0.6 + 0.4 * np.sqrt(np.clip(x, 0, 1)))
        if on("print"):  # dot gain + limited tonal range + coarse quantisation (print/scan proxy)
            x = cv2.GaussianBlur(x, (0, 0), 0.6 + 0.8 * s); x = 0.08 + 0.84 * np.clip(x, 0, 1) ** (1 + 0.3 * s)
            x = np.round(x * 7) / 7
        if on("jpeg"):
            q = int(np.clip(95 - 55 * s * (r.rand() if only is None else 1), 20, 95))
            _, buf = cv2.imencode(".jpg", (np.clip(x, 0, 1) * 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
            x = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        out[i] = np.clip(x, 0, 1)
    return out


LEVELS = {"digital": 0.0, "L1": 0.33, "L2": 0.66, "L3": 1.0}


# ---------------------------------------------------------------- instances
def load_instance(name, runs=RUNS):
    d = os.path.join(runs, name); cfg = json.load(open(os.path.join(d, "config.json")))
    e, dc = Encoder(cfg.get("width", 1)), Decoder(cfg.get("width", 1))
    e.load_state_dict(torch.load(os.path.join(d, "enc.pt"))); dc.load_state_dict(torch.load(os.path.join(d, "dec.pt")))
    return e.eval(), dc.eval(), cfg


@torch.no_grad()
def render(enc, bits, bs=500):
    """Digital visual code as 8-bit image file would store it, (N,H,W) float32."""
    out = [enc(torch.from_numpy(bits[i:i + bs]).float()) for i in range(0, len(bits), bs)]
    return (torch.cat(out)[:, 0].numpy() * 255).round().astype(np.float32) / 255


@torch.no_grad()
def decode(dec, imgs, bs=500):
    out = [dec(torch.from_numpy(imgs[i:i + bs, None])) for i in range(0, len(imgs), bs)]
    return (torch.cat(out) > 0).numpy().astype(np.uint8)


# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def metrics(pred, true, boot=300, seed=0):
    """BER (bit errors/32), exact-message rate (all 32 bits), CRC-accept rate (FAR/FRR basis)."""
    err = (pred != true); per = err.mean(1); n = len(true)
    ex = (err.sum(1) == 0); acc = crc_ok(pred)
    rng = np.random.RandomState(seed); bs = [per[rng.randint(0, n, n)].mean() for _ in range(boot)]
    return dict(ber=float(per.mean()), ber_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                exact=float(ex.mean()), exact_ci=list(wilson(int(ex.sum()), n)),
                crc_accept=float(acc.mean()), crc_accept_ci=list(wilson(int(acc.sum()), n)),
                accepted_and_wrong=float((acc & ~ex).mean()), n=n)


def sha256(path):
    h = hashlib.sha256(); h.update(open(path, "rb").read()); return h.hexdigest()


def tci(vals, z=None):
    """mean and 95% t-interval across independent instances."""
    from scipy import stats
    v = np.asarray(vals, float); n = len(v)
    if n < 2: return float(v.mean()), [float("nan")] * 2
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return float(v.mean()), [float(v.mean() - h), float(v.mean() + h)]
