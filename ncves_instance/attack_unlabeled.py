"""Experiment D (unlabeled): attacker observes only codes of the ORIGINAL encoder (no messages).
Learns an unsupervised representation (PCA, or autoencoder), then optionally fits a linear read-out with K labels.
K=0 has no defined read-out: bits are the sign of latent dims (arbitrary alignment) -> reported as-is.
Diagnostic only: max |corr| between latent dims and true bits on test (oracle; not an attack)."""
import json, os, numpy as np, torch
from common import *
torch.set_num_threads(3); torch.manual_seed(5)
S = make_splits(); enc, _, _ = load_instance("orig")
U = render(enc, S["attacker"]); test = S["test"]; tc = render(enc, test)  # unlabeled observed codes
conds = {"digital": 0.0, "L2": 0.66}
degT = {c: np_channel(tc, s, 11) for c, s in conds.items()}
Utr = np_channel(U[:10000], 0.33, 5)  # attacker observes slightly degraded codes (photos)


def probe(Ztr, Y, Zte, ridge=1e-2):
    A = np.c_[Ztr, np.ones(len(Ztr))]; W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ (Y * 2 - 1.0))
    return (np.c_[Zte, np.ones(len(Zte))] @ W > 0).astype(np.uint8)


def oracle_corr(Z, Y):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9); Ys = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    C = np.abs(Zs.T @ Ys / len(Z)); return float(C.max(0).mean())  # per true bit, best latent dim


# --- PCA latent
flat = lambda x: x.reshape(len(x), -1)
mu = flat(Utr).mean(0); _, _, Vt = np.linalg.svd(flat(Utr)[:5000] - mu, full_matrices=False)
pca = lambda x, k=64: (flat(x) - mu) @ Vt[:k].T
# --- autoencoder latent (reuse Decoder as encoder -> 64d, Encoder-style decoder)
class AE(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.e = Decoder(); s.e.net[-1] = torch.nn.Linear(256, 64)
        s.d = Encoder(); s.d.fc[0] = torch.nn.Linear(64, 512)
    def forward(s, x): z = s.e(x); return z, s.d.up(s.d.fc(z).view(-1, s.d.c, 8, 8)).sigmoid()
ae = AE(); opt = torch.optim.Adam(ae.parameters(), 1e-3); ch = DiffChannel(); Ut = torch.from_numpy(U)[:, None]
for step in range(2500):
    x = Ut[torch.randint(0, len(Ut), (128,))]; s = torch.rand(128) * 0.5
    _, r = ae(ch(x, s)); loss = torch.nn.functional.mse_loss(r, x); opt.zero_grad(); loss.backward(); opt.step()
ae.eval()
@torch.no_grad()
def aez(x): return torch.cat([ae(torch.from_numpy(x[i:i + 500, None]))[0] for i in range(0, len(x), 500)]).numpy()

pool = S["attacker"]; Ktr = {"pca": lambda x: pca(x), "ae": aez}
res = {}
for rep, f in Ktr.items():
    Ztr_all = f(np_channel(U[:2000], 0.33, 6)); res[rep] = {}
    for K in [0, 32, 100, 1000, 2000]:
        for c in conds:
            Zte = f(degT[c])
            if K == 0:  # no labels: sign of first 32 centred latent dims, arbitrary alignment
                pred = (Zte[:, :32] > Ztr_all[:, :32].mean(0)).astype(np.uint8)
            else:
                pred = probe(Ztr_all[:K], pool[:K], Zte)
            res[rep][f"K{K}_{c}"] = metrics(pred, test)
    res[rep]["oracle_corr_digital"] = oracle_corr(f(degT["digital"]), test)
json.dump(res, open(f"{RES}/attack_unlabeled.json", "w"), indent=1)
for rep in res:
    print(rep, {k: round(v["ber"], 3) for k, v in res[rep].items() if k != "oracle_corr_digital"}, "oracle corr", round(res[rep]["oracle_corr_digital"], 3))
