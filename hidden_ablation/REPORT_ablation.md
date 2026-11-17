# NCVES vs HiDDeN-style joint training: controlled ablation

Earlier results (`../ncves_instance/`) are untouched. All numbers below come from `results/eval_h.json`, `results/attack_summary.md`, `results/attacks/*.json`.
Setup: 32-bit payload (24 info + CRC-8), 64×64 grayscale, same held-out 2000 messages for all models, 3000 training steps per model, CPU.
Channels: digital, L2 (simulated blur/noise/perspective/illumination/JPEG; **no physical print-and-photograph test was done**).
Metrics were fixed before the analysis: (1) diagonal BER/exact, (2) off-diagonal BER/exact, (3) substitute-decoder BER/exact at N=25/100/500 under identical protocol.

## 1. Source audit: what NCVES actually is
- The repo's `train.py` has no student–teacher mechanism. Its only distinct components are an explicit key input to the encoder and a negated-BCE key-separation loss. As written that loss never trained (BER 0.47 with right key and wrong key; see `../ncves_instance/results/audit_original.json`).
- The "instance-only NCVES" (Group A) used in the earlier study is ordinary joint encoder–decoder training through a differentiable channel (BCE message loss). No component distinguishes it from conventional joint training, apart from the lack of a cover image.
- HiDDeN (Zhu et al., ECCV 2018; only the arXiv abstract was consulted, not the full text/code): jointly trained encoder/decoder, cover image, MSE image loss, adversarial discriminator, distortion layers (blur, dropout, crop, JPEG approximation). Its claims concern imperceptibility and robustness, not secrecy against independently trained decoders. Independently trained HiDDeN models almost certainly also fail to cross-decode (random initialization), but **this was not shown with a faithful HiDDeN reproduction**.
- Genuine algorithmic differences between NCVES(A) and HiDDeN: none identified other than absence of cover/discriminator; the difference is in what is claimed (instance-specificity as an intended property), not in the training algorithm.

## 2. Groups (what was actually executed)
| Group | Content | Models | Status |
|---|---|---|---|
| A | existing NCVES instances (orig, B_r1–5), reused | 6 | complete (reused) |
| B | **adapted** HiDDeN-style (cover-conditioned, BCE + 0.7·MSE, no adversary, 9.7k-param encoder, synthetic covers) – NOT a faithful reproduction | H0–H4 (5) | complete; **fails to converge** |
| B2 | same as B with λ_img=0.05 | HL0–2 (3) | complete; **fails to converge** |
| C | repo-style key-conditioned encoder, key0 = no separation loss, key1 = bounded separation loss (the only identifiable "mechanism") | 2 + 2 | complete (2 seeds each) |
| D | independent initialization | = off-diagonal entries of all matrices | complete |
Faithful HiDDeN (with adversary, original capacity, real images): **unexecuted**. Fixed-code/QR baseline: not repeated here (a fixed grid was tested earlier; any decoder reads it).

## 3. Cross-decoding (BER; full matrices in `results/eval_h.json`)
| Group | channel | pairs | diagonal BER | off-diag BER | gap [95% t-CI over pairs] | diag exact | off exact (max) |
|---|---|---|---|---|---|---|---|
| A NCVES | digital | 6 | 0.0000 | 0.4956 | 0.496 [0.492, 0.499] | ≈1.00 | 0 |
| A NCVES | L2 | 6 | 0.0001 | 0.4954 | 0.495 [0.492, 0.498] | ≈1.00 | 0 |
| B HiDDeN-style | digital | 5 | 0.238±0.019 | 0.499 | 0.260 [0.238, 0.283] | 0.000 | 0 |
| B HiDDeN-style | L2 | 5 | 0.371±0.008 | 0.498 | 0.127 [0.115, 0.138] | 0.000 | 0 |
| B2 low λ_img | digital | 3 | 0.231±0.022 | 0.501 | 0.270 [0.207, 0.332] | 0.0002 | 0 |
| B2 low λ_img | L2 | 3 | 0.383±0.008 | 0.505 | 0.122 [0.098, 0.145] | 0 | 0 |
| C key0 | digital | 2 | 0.0000 | 0.496 | 0.496 (CI n/a, n=2) | 1.000 | 0 |
| C key0 | L2 | 2 | 0.0001 | 0.505 | 0.505 | 0.996 | 0 |
| C key1 | digital | 2 | 0.0000 | 0.497 | 0.497 | 1.000 | 0 |
| C key1 | L2 | 2 | 0.0003 | 0.496 | 0.496 | 0.992 | 0 |
Group A exact accuracy for the matrix was not stored in the earlier `evaluate.json` (BER only); its exact values are in REPORT.md of the earlier study. Gap CIs use one value per pair (paired over the same messages); bits are not treated as independent. Group A diag/off orientation was taken as stored.

**Key finding:** the HiDDeN-style baseline never learned to decode its own codes (diag BER 0.23–0.38, exact ≈0). Its gap is therefore not a cross-decoding "secrecy" effect; it is just an under-trained system. The comparison "NCVES gap > HiDDeN gap" is **confounded by failed convergence and capacity mismatch (9.7k vs 1.08M params)** and must not be read as an NCVES advantage. Where both systems do decode (A, C), the off-diagonal result is identical: chance (BER ≈0.50, exact 0, CRC accept ≈ chance 0.0039).

## 4. Substitute-decoder attack (same protocol, 3 attacker seeds, 3000 steps, fresh decoder, labeled pairs from disjoint pool)
| target | N | clean BER | clean exact | L2 BER | L2 exact |
|---|---|---|---|---|---|
| NCVES orig (reused) | 25 | 0.203 | 0.001 | 0.208 | 0.001 |
| NCVES orig | 100 | 0.0034 | 0.902 | 0.0072 | 0.806 |
| NCVES orig | 500 | 0.0000 | 1.000 | 0.0001 | 0.998 |
| C key0 (no sep. loss) | 100 | 0.0028 | 0.917 | 0.0076 | 0.793 |
| C key1 (sep. loss) | 100 | 0.0032 | 0.902 | 0.0089 | 0.762 |
| B H0 (HiDDeN-style) | 25/100/500 | 0.449/0.406/0.358 | 0 | 0.458/0.433/0.414 | 0 |
H0 numbers are **not** evidence of attack resistance: the attacker is evaluated on codes that even the legitimate decoder cannot read (it carries little message information). Attacks on a converged conventional baseline were **not executed** because no converged baseline was obtained. The only valid like-for-like attack comparison is NCVES vs key0 vs key1: all are broken by ~100 labeled pairs, identically within seed noise.

## 5. Mechanism ablation (Group C)
- Removing the separation loss (key0 vs key1) changes neither diagonal accuracy, independent cross-decoding (chance in both), nor substitute-attack success (≈0.90 exact at N=100 in both).
- It does change one thing: wrong *random* keys fed to the own encoder. key0: decoder still reads the code (BER 0.001–0.008, i.e. key is ignored); key1: BER 0.32/0.53 digital (K1_0 partial, K1_1 full separation). But a **one-bit-flipped key still decodes with BER 0.000 in all four models**, so the loss gives no fine-grained key dependence. This is an explicit-key effect (2 seeds), not an instance effect, and is the same family as key-controlled data hiding.
- Independent-instance failure occurs with or without any NCVES-specific component (key0, Group A, and Group B all fail across instances).

## 6. Conclusion
**Outcome 1 — no demonstrated distinction.** Cross-decoding failure across independently initialized models is reproduced in every group, including the plain-joint-training baselines with no NCVES-specific component (key0) and is therefore explained by ordinary joint training + random initialization. Substitute decoders defeat NCVES, key0 and key1 equally with ~100 labels. The HiDDeN-style comparison is inconclusive/invalid because it did not converge (a faithful reproduction remains unexecuted), so the data cannot even say whether NCVES differs from true HiDDeN; they show only that the claimed mechanism has no separable component beyond joint training. No technically defensible distinction over HiDDeN-style joint training has been demonstrated, and this says nothing about patentability. Highest-value next experiment: a converged, faithful (or capacity-matched) HiDDeN-type baseline, then the identical 25/100/500 attack on it.

## 7. Limitations
Adapted/non-converged baseline; capacity mismatch; synthetic covers; 2 seeds for C (CI not computable); 3 attacker seeds; simulated channels only; no physical tests; HiDDeN paper consulted only via abstract; machine load varied (affects latency only; decode ≈0.37 ms/code on CPU).

## 8. Commands (from `experiments/hidden_ablation/`)
```
./run_train_h.sh                                   # H0-H4, K0_0/1, K1_0/1  (logs/train.log)
python train_h.py --name HL{0,1,2} --kind hidden --seed {0,1,2} --lam_img 0.05 --threads 2
./run_attacks_h.sh                                 # results/attacks/*.json
python summarize_attacks.py                        # results/attack_summary.md
python eval_h.py                                   # results/eval_h.json
```
Checkpoints: `runs/*/{enc,dec}.pt`, `config.json`, `log.jsonl`. Aborted larger attempt kept in `_aborted_slow_hidden_w32/`.
