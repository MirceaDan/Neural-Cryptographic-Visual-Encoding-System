"""Experiment 4 + 6: fingerprints, weight-level vs behavioural differences, weight-theft consequences,
decoder-only forging, and the (optional, NOT primary) conventional-key XOR baseline."""
import json, os, numpy as np, torch
from common import *
torch.set_num_threads(3); torch.manual_seed(0); np.random.seed(0)
S = make_splits(); test, pool = S["test"], S["attacker"]
names = sorted(n for n in os.listdir(RUNS) if os.path.exists(f"{RUNS}/{n}/config.json"))
inst = {n: load_instance(n) for n in names}; R = {}
codes = {n: render(inst[n][0], test) for n in names}
flat = lambda m: torch.cat([p.detach().flatten() for p in m.parameters()])
ber = lambda dec, x, y=test: float((decode(dec, x) != y).mean())
L2 = lambda x: np_channel(x, 0.66, 11)
D0, E0 = inst["orig"][1], inst["orig"][0]

# ---- 1. weights vs behaviour
w0 = flat(D0); e0 = flat(E0)
R["weight_distance"] = {n: dict(dec_rel_l2=float((flat(inst[n][1]) - w0).norm() / w0.norm()) if len(flat(inst[n][1])) == len(w0) else None,
                                dec_cosine=float(torch.cosine_similarity(flat(inst[n][1]), w0, 0)) if len(flat(inst[n][1])) == len(w0) else None,
                                enc_rel_l2=float((flat(inst[n][0]) - e0).norm() / e0.norm()) if len(flat(inst[n][0])) == len(e0) else None,
                                ber_on_orig_codes_L2=ber(inst[n][1], L2(codes["orig"]))) for n in names if n != "orig"}
# noise sensitivity of the ORIGINAL decoder (how "exact" must the weights be?)
import copy
x_dig, x_L2 = codes["orig"], L2(codes["orig"]); R["weight_noise"] = {}
for eps in [0, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]:
    r = []
    for t in range(3):
        d = copy.deepcopy(D0); g = torch.Generator().manual_seed(t)
        with torch.no_grad():
            for p in d.parameters(): p.add_(eps * p.std() * torch.randn(p.shape, generator=g)) if p.numel() > 1 else None
        r.append((ber(d, x_dig), ber(d, x_L2)))
    R["weight_noise"][str(eps)] = dict(digital=float(np.mean([a for a, _ in r])), L2=float(np.mean([b for _, b in r])))
# linear interpolation orig -> independent retrain B_r1 (same architecture)
R["interpolation_to_B_r1"] = {}
for al in [0, .1, .25, .5, .75, .9, 1]:
    d = copy.deepcopy(D0); s1 = inst["B_r1"][1].state_dict()
    d.load_state_dict({k: (1 - al) * v + al * s1[k] for k, v in D0.state_dict().items()}); R["interpolation_to_B_r1"][str(al)] = ber(d, x_dig)

# ---- 2. fingerprints: identification vs necessity
ids = [n for n in names if inst[n][2].get("width", 1) == 1]
Ptr = {n: render(inst[n][0], pool[:1500]) for n in ids}
mu = {n: Ptr[n].mean(0) for n in ids}
clf = Decoder(); clf.net[-1] = torch.nn.Linear(256, len(ids)); opt = torch.optim.Adam(clf.parameters(), 1e-3); ch = DiffChannel()
X = torch.from_numpy(np.stack([Ptr[n] for n in ids]))  # (K,1500,H,W)
for step in range(800):
    k = torch.randint(0, len(ids), (128,)); j = torch.randint(0, 1500, (128,))
    x = ch(X[k, j][:, None], torch.rand(128) * 0.7)
    l = torch.nn.functional.cross_entropy(clf(x), k); opt.zero_grad(); l.backward(); opt.step()
clf.eval()
def acc(level):
    ok = []
    for i, n in enumerate(ids):
        x = np_channel(codes[n], level, 21); ok.append((clf(torch.from_numpy(x[:, None])).argmax(1) == i).float().mean().item())
    return float(np.mean(ok))
R["fingerprint_identification"] = dict(n_classes=len(ids), chance=1 / len(ids), acc_digital=acc(0.0), acc_L2=acc(0.66))
Dn, En = D0, E0
sw = {}
for tgt in ["B_r1", "B_r2"]:
    I = codes["orig"]
    sw[f"orig_code_minus_mu_orig_plus_mu_{tgt}"] = ber(D0, np.clip(I - mu["orig"] + mu[tgt], 0, 1))
    sw[f"{tgt}_code_plus_fingerprint_shift_to_orig_decoded_by_orig"] = ber(D0, np.clip(codes[tgt] - mu[tgt] + mu["orig"], 0, 1))
sw["orig_code_unchanged"] = ber(D0, codes["orig"])
sw["orig_code_fingerprint_mean_removed(const)"] = ber(D0, np.clip(codes["orig"] - mu["orig"] + mu["orig"].mean(), 0, 1))
sw["B_r1_code_decoded_by_orig"] = ber(D0, codes["B_r1"])
sw["B_r1_code_decoded_by_orig_after_mean_shift"] = ber(D0, np.clip(codes["B_r1"] - mu["B_r1"] + mu["orig"], 0, 1))
R["fingerprint_necessity"] = sw
cc = lambda a, b: float(np.mean([np.corrcoef(a[i].ravel(), b[i].ravel())[0, 1] for i in range(300)]))
cm = lambda a: a - a.mean(0)
R["code_similarity_same_message"] = {n: cc(cm(codes["orig"]), cm(codes[n])) for n in ids if n != "orig"}
R["code_similarity_diff_message_same_instance"] = cc(cm(codes["orig"]), np.roll(cm(codes["orig"]), 1, 0))
R["mean_fingerprint_energy_fraction"] = {n: float((mu[n] - mu[n].mean()).var() / codes[n].var()) for n in ids}

# ---- 3. weight theft: decoder-only forging by gradient inversion (no encoder access)
m = torch.from_numpy(test[:300]).float(); lg = torch.zeros(300, 1, IMG, IMG, requires_grad=True)
opt = torch.optim.Adam([lg], 0.05); bce = torch.nn.BCEWithLogitsLoss()
for p in D0.parameters(): p.requires_grad_(False)
for step in range(400):
    l = bce(D0(ch(torch.sigmoid(lg), torch.rand(300) * 0.7)), m); opt.zero_grad(); l.backward(); opt.step()
forged = (torch.sigmoid(lg).detach()[:, 0].numpy() * 255).round() / 255
tt = test[:300]
R["decoder_only_forgery"] = dict(
    orig_decodes_digital=ber(D0, forged, tt), orig_decodes_L2=ber(D0, L2(forged), tt),
    B_r1_decodes_forgery_digital=ber(inst["B_r1"][1], forged, tt), B_r2_decodes_forgery_digital=ber(inst["B_r2"][1], forged, tt),
    pixel_mse_to_genuine_code=float(((forged - codes["orig"][:300]) ** 2).mean()),
    classified_as_orig_fraction=float((clf(torch.from_numpy(forged[:, None])).argmax(1) == ids.index("orig")).float().mean()))

# ---- 4. optional conventional-key baseline (separate from the primary hypothesis)
rk = np.random.RandomState(9); k = rk.randint(0, 2, NBITS).astype(np.uint8); k2 = rk.randint(0, 2, NBITS).astype(np.uint8)
xk = L2(render(E0, test ^ k)); R["xor_key_baseline_NOT_primary"] = dict(
    note="msg XOR k before encoding; attacker holds full original instance but not k",
    right_key_ber=ber(D0, xk, test ^ k) , right_key_after_unxor=float(((decode(D0, xk) ^ k) != test).mean()),
    wrong_key_ber=float(((decode(D0, xk) ^ k2) != test).mean()))
json.dump(R, open(f"{RES}/analysis.json", "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k not in ("weight_distance",)}, indent=1)[:6000])
