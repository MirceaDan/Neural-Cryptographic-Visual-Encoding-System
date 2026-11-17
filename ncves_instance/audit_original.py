"""Runs the UNMODIFIED repo train.py objective (imported) and measures BER with
correct vs. wrong key. Shows the key-separation objective is ill-posed."""
import sys, json, torch
sys.path.insert(0, "../..")
import train as T
torch.manual_seed(0)
m = T.NCVES(); opt = torch.optim.Adam(m.parameters(), lr=1e-4)
log = []
for step in range(1000):
    msg = torch.randint(0, 2, (32, 128)).float()
    k, wk = (torch.randint(0, 2, (32, 64)).float() for _ in range(2))
    loss = T.total_loss(m, msg, k, wk)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0: log.append((step, loss.item()))
with torch.no_grad():
    msg = torch.randint(0, 2, (2000, 128)).float()
    k, wk = (torch.randint(0, 2, (2000, 64)).float() for _ in range(2))
    ber = lambda key: ((m(msg, key)[2] > 0).float() != msg).float().mean().item()
    out = dict(loss_log=log, ber_correct_key=ber(k), ber_wrong_key=ber(wk),
               ber_same_key_as_each_other=ber(k.clone()))
print(json.dumps(out, indent=1)); json.dump(out, open("results/audit_original.json", "w"), indent=1)
