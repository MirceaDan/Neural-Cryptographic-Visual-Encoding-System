import argparse, json, os, sys, time, numpy as np, torch
from lib_h import *
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--kind", required=True, choices=["hidden", "key0", "key1"])
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--lam_img", type=float, default=0.7); ap.add_argument("--lam_key", type=float, default=1.0)
a = ap.parse_args(); torch.set_num_threads(a.threads); out = f"{RUNS_H}/{a.name}"
if os.path.exists(out): sys.exit("refusing to overwrite")
os.makedirs(out); S = make_splits(); train = torch.from_numpy(S["train"]).float(); cv_tr = torch.from_numpy(covers()["train"])
torch.manual_seed(a.seed); enc = HEnc() if a.kind == "hidden" else KEnc(); dec = Decoder(); torch.manual_seed(a.seed + 10000)
ch = DiffChannel(); params = list(enc.parameters()) + list(dec.parameters()); lr = 1e-3
opt = torch.optim.Adam(params, lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=a.steps, pct_start=0.05)
rs = np.random.RandomState(a.seed); bce = torch.nn.BCEWithLogitsLoss(); key = secret_key(a.seed); t0 = time.time(); lg = open(f"{out}/log.jsonl", "w")
for step in range(a.steps):
    idx = torch.from_numpy(rs.randint(0, len(train), 128)); m = train[idx]
    s = torch.rand(128) * min(1., .15 + .85 * step / (.4 * a.steps)) * (torch.rand(128) > .1)
    if a.kind == "hidden":
        cov = cv_tr[idx][:, None]; x = enc(m, cov); li = torch.nn.functional.mse_loss(x, cov); lm = bce(dec(ch(x, s)), m); loss = lm + a.lam_img * li
    else:
        x = enc(m, key.expand(128, -1)); lm = bce(dec(ch(x, s)), m); li = torch.zeros(()); loss = lm
        if a.kind == "key1":
            wk = torch.randint(0, 2, (128, KEYBITS)).float(); wk = torch.where((wk == key).all(1, keepdim=True), 1 - wk, wk)
            ls = bce(dec(ch(enc(m, wk), s)), torch.full_like(m, 0.5)); loss = loss + a.lam_key * ls  # bounded separation: wrong-key output -> uninformative
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 100 == 0 or step == a.steps - 1: lg.write(json.dumps(dict(step=step, msg_loss=lm.item(), img_mse=li.item(), loss=loss.item())) + "\n"); lg.flush()
torch.save(enc.state_dict(), f"{out}/enc.pt"); torch.save(dec.state_dict(), f"{out}/dec.pt")
cfg = vars(a) | dict(init_seed=a.seed, train_seconds=time.time() - t0, enc_params=nparams(enc), dec_params=nparams(dec))
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1); print(a.name, "done", f"{time.time()-t0:.0f}s", flush=True)
