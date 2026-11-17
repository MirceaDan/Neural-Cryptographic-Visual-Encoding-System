Cell format: BER [95% bootstrap CI over 2000 test messages] / exact-message rate / CRC-accept rate.
For the original decoder CRC-accept = 1-FRR; for any unauthorised decoder CRC-accept = FAR(CRC). Chance: BER 0.5, exact 2^-32, CRC-accept 2^-8=0.0039.

## Table 1. Decoders applied to held-out codes from the ORIGINAL encoder

| scenario | decoder | digital | L2 | L3 |
|---|---|---|---|---|
| A legitimate | orig | 0.0000 [0.0000,0.0000] / 1.000 / 1.000 | 0.0000 [0.0000,0.0001] / 0.999 / 0.999 | 0.0030 [0.0024,0.0036] / 0.944 / 0.944 |
| B independent retrain | B_r1 | 0.4940 [0.4905,0.4975] / 0.000 / 0.004 | 0.4951 [0.4915,0.4991] / 0.000 / 0.006 | 0.4975 [0.4943,0.5014] / 0.000 / 0.006 |
| B independent retrain | B_r2 | 0.4950 [0.4912,0.4983] / 0.000 / 0.004 | 0.5002 [0.4965,0.5042] / 0.000 / 0.004 | 0.4991 [0.4957,0.5028] / 0.000 / 0.005 |
| B independent retrain | B_r3 | 0.5053 [0.5005,0.5086] / 0.000 / 0.002 | 0.5017 [0.4974,0.5058] / 0.000 / 0.003 | 0.5013 [0.4975,0.5051] / 0.000 / 0.002 |
| B independent retrain | B_r4 | 0.4915 [0.4883,0.4952] / 0.000 / 0.004 | 0.4952 [0.4918,0.4987] / 0.000 / 0.003 | 0.4966 [0.4933,0.5010] / 0.000 / 0.004 |
| B independent retrain | B_r5 | 0.4770 [0.4734,0.4809] / 0.000 / 0.003 | 0.4743 [0.4704,0.4782] / 0.000 / 0.004 | 0.4757 [0.4721,0.4795] / 0.000 / 0.007 |
| **B aggregate (n=5)** | mean, 95% t-CI across instances | BER 0.4925 [0.4799,0.5052] / exact max 0.0000 / CRC-acc mean 0.0032 | BER 0.4933 [0.4796,0.5069] / exact max 0.0000 / CRC-acc mean 0.0038 | BER 0.4940 [0.4811,0.5070] / exact max 0.0000 / CRC-acc mean 0.0050 |
| C variation | C_batch64 | 0.4985 [0.4943,0.5019] / 0.000 / 0.004 | 0.4919 [0.4879,0.4955] / 0.000 / 0.005 | 0.4914 [0.4877,0.4954] / 0.000 / 0.003 |
| C variation | C_diffinit_samedata | 0.4958 [0.4919,0.5000] / 0.000 / 0.005 | 0.4977 [0.4938,0.5016] / 0.000 / 0.005 | 0.4990 [0.4948,0.5028] / 0.000 / 0.004 |
| C variation | C_lr2x | 0.4876 [0.4836,0.4909] / 0.000 / 0.002 | 0.4874 [0.4829,0.4910] / 0.000 / 0.007 | 0.4842 [0.4800,0.4877] / 0.000 / 0.004 |
| C variation | C_lrhalf | 0.5031 [0.4996,0.5072] / 0.000 / 0.004 | 0.5053 [0.5011,0.5094] / 0.000 / 0.004 | 0.5029 [0.4990,0.5069] / 0.000 / 0.003 |
| C variation | C_onlynoise_seed | 0.0428 [0.0412,0.0444] / 0.269 / 0.269 | 0.0573 [0.0552,0.0591] / 0.167 / 0.169 | 0.0780 [0.0756,0.0804] / 0.102 / 0.104 |
| C variation | C_sameinit_diffdata | 0.0671 [0.0648,0.0692] / 0.142 / 0.144 | 0.0824 [0.0801,0.0849] / 0.086 / 0.088 | 0.1061 [0.1032,0.1089] / 0.049 / 0.051 |
| C variation | C_steps1500 | 0.4934 [0.4895,0.4975] / 0.000 / 0.005 | 0.4954 [0.4916,0.4995] / 0.000 / 0.002 | 0.4967 [0.4929,0.5011] / 0.000 / 0.004 |
| C variation | C_width2 | 0.4909 [0.4871,0.4947] / 0.000 / 0.005 | 0.4895 [0.4855,0.4931] / 0.000 / 0.001 | 0.4879 [0.4843,0.4914] / 0.000 / 0.003 |
| D attacker, labeled pairs | sup_n25 (n_seeds=3) | BER 0.2029 [0.1870,0.2188] / exact max 0.0010 / CRC-acc 0.0080 | BER 0.2076 [0.1908,0.2245] / exact max 0.0015 / CRC-acc 0.0077 | BER 0.2169 [0.2045,0.2293] / exact max 0.0010 / CRC-acc 0.0053 |
| D attacker, labeled pairs | sup_n100 (n_seeds=3) | BER 0.0034 [0.0021,0.0046] / exact max 0.9155 / CRC-acc 0.9017 | BER 0.0072 [0.0058,0.0085] / exact max 0.8155 / CRC-acc 0.8063 | BER 0.0158 [0.0153,0.0163] / exact max 0.6645 / CRC-acc 0.6573 |
| D attacker, labeled pairs | sup_n500 (n_seeds=3) | BER 0.0000 [0.0000,0.0000] / exact max 1.0000 / CRC-acc 1.0000 | BER 0.0001 [0.0000,0.0001] / exact max 0.9990 / CRC-acc 0.9983 | BER 0.0029 [0.0025,0.0034] / exact max 0.9400 / CRC-acc 0.9373 |
| D attacker, labeled pairs | sup_n2500 (n_seeds=3) | BER 0.0000 [0.0000,0.0000] / exact max 1.0000 / CRC-acc 1.0000 | BER 0.0001 [0.0000,0.0001] / exact max 0.9990 / CRC-acc 0.9987 | BER 0.0025 [0.0019,0.0032] / exact max 0.9535 / CRC-acc 0.9502 |
| D attacker, labeled pairs | sup_n20000 (n_seeds=3) | BER 0.0000 [0.0000,0.0000] / exact max 1.0000 / CRC-acc 1.0000 | BER 0.0001 [0.0000,0.0001] / exact max 0.9990 / CRC-acc 0.9987 | BER 0.0025 [0.0021,0.0030] / exact max 0.9505 / CRC-acc 0.9495 |
| D attacker, labeled pairs | sup_n25|fromB_r1 (n_seeds=1) | BER 0.1995 [nan,nan] / exact max 0.0015 / CRC-acc 0.0045 | BER 0.2050 [nan,nan] / exact max 0.0015 / CRC-acc 0.0050 | BER 0.2151 [nan,nan] / exact max 0.0005 / CRC-acc 0.0045 |
| D attacker, labeled pairs | sup_n100|fromB_r1 (n_seeds=1) | BER 0.0036 [nan,nan] / exact max 0.8950 / CRC-acc 0.8950 | BER 0.0073 [nan,nan] / exact max 0.8035 / CRC-acc 0.8035 | BER 0.0185 [nan,nan] / exact max 0.6360 / CRC-acc 0.6360 |
| D attacker, labeled pairs | sup_n500|fromB_r1 (n_seeds=1) | BER 0.0000 [nan,nan] / exact max 0.9995 / CRC-acc 0.9995 | BER 0.0004 [nan,nan] / exact max 0.9905 / CRC-acc 0.9905 | BER 0.0050 [nan,nan] / exact max 0.9005 / CRC-acc 0.9005 |
| D attacker, labeled pairs | sup_n2500|w2 (n_seeds=1) | BER 0.0000 [nan,nan] / exact max 1.0000 / CRC-acc 1.0000 | BER 0.0001 [nan,nan] / exact max 0.9985 / CRC-acc 0.9985 | BER 0.0024 [nan,nan] / exact max 0.9525 / CRC-acc 0.9525 |
| D unlabeled codes + K labels | pca, K=0 | 0.5002 [0.4971,0.5039] / 0.000 / 0.004 | 0.4957 [0.4922,0.4996] / 0.000 / 0.004 | n/a |
| D unlabeled codes + K labels | pca, K=32 | 0.0234 [0.0221,0.0248] / 0.522 / 0.522 | 0.0565 [0.0539,0.0590] / 0.247 / 0.248 | n/a |
| D unlabeled codes + K labels | pca, K=100 | 0.0000 [0.0000,0.0000] / 1.000 / 1.000 | 0.0134 [0.0120,0.0147] / 0.767 / 0.767 | n/a |
| D unlabeled codes + K labels | pca, K=1000 | 0.0000 [0.0000,0.0000] / 1.000 / 1.000 | 0.0004 [0.0002,0.0006] / 0.989 / 0.989 | n/a |
| D unlabeled codes + K labels | pca, K=2000 | 0.0000 [0.0000,0.0000] / 1.000 / 1.000 | 0.0004 [0.0002,0.0005] / 0.988 / 0.988 | n/a |
| D unlabeled codes + K labels | ae, K=0 | 0.5076 [0.5033,0.5115] / 0.000 / 0.004 | 0.5081 [0.5041,0.5123] / 0.000 / 0.004 | n/a |
| D unlabeled codes + K labels | ae, K=32 | 0.1181 [0.1134,0.1232] / 0.230 / 0.231 | 0.1235 [0.1187,0.1289] / 0.202 / 0.205 | n/a |
| D unlabeled codes + K labels | ae, K=100 | 0.0000 [0.0000,0.0000] / 1.000 / 1.000 | 0.0003 [0.0002,0.0005] / 0.992 / 0.992 | n/a |
| D unlabeled codes + K labels | ae, K=1000 | 0.0000 [0.0000,0.0000] / 1.000 / 1.000 | 0.0000 [0.0000,0.0001] / 0.999 / 0.999 | n/a |
| D unlabeled codes + K labels | ae, K=2000 | 0.0000 [0.0000,0.0000] / 1.000 / 1.000 | 0.0000 [0.0000,0.0001] / 0.999 / 0.999 | n/a |

## Table 2. Single-factor physical-channel simulation (s=1), BER: original vs B mean (n=5)

| decoder | persp@1 | blur@1 | illum@1 | noise@1 | jpeg@1 | occl@1 | print@1 |
|---|---|---|---|---|---|---|---|
| orig | 0.0135 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| B mean | 0.4949 | 0.4921 | 0.4925 | 0.4923 | 0.4923 | 0.4924 | 0.4924 |
| orig exact | 0.837 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 |

## Table 3. Cross matrix BER, digital (rows: encoder that made the codes; cols: decoder)

| enc \ dec | B_r1 | B_r2 | B_r3 | B_r4 | B_r5 | C_batch64 | C_diffinit_samedata | C_lr2x | C_lrhalf | C_onlynoise_seed | C_sameinit_diffdata | C_steps1500 | C_width2 | orig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_r1 | 0.000 | 0.501 | 0.502 | 0.470 | 0.514 | 0.514 | 0.495 | 0.489 | 0.494 | 0.498 | 0.511 | 0.494 | 0.498 | 0.495 |
| B_r2 | 0.503 | 0.000 | 0.501 | 0.494 | 0.502 | 0.500 | 0.501 | 0.503 | 0.498 | 0.501 | 0.495 | 0.490 | 0.489 | 0.493 |
| B_r3 | 0.496 | 0.499 | 0.000 | 0.494 | 0.487 | 0.493 | 0.486 | 0.507 | 0.498 | 0.502 | 0.492 | 0.492 | 0.487 | 0.494 |
| B_r4 | 0.472 | 0.500 | 0.495 | 0.000 | 0.495 | 0.489 | 0.488 | 0.514 | 0.502 | 0.498 | 0.522 | 0.499 | 0.504 | 0.498 |
| B_r5 | 0.508 | 0.511 | 0.507 | 0.496 | 0.000 | 0.499 | 0.516 | 0.497 | 0.508 | 0.487 | 0.476 | 0.494 | 0.502 | 0.477 |
| C_batch64 | 0.518 | 0.504 | 0.490 | 0.484 | 0.499 | 0.000 | 0.486 | 0.484 | 0.483 | 0.500 | 0.501 | 0.496 | 0.492 | 0.498 |
| C_diffinit_samedata | 0.497 | 0.490 | 0.481 | 0.498 | 0.514 | 0.491 | 0.000 | 0.499 | 0.510 | 0.511 | 0.512 | 0.490 | 0.508 | 0.494 |
| C_lr2x | 0.485 | 0.506 | 0.514 | 0.507 | 0.482 | 0.486 | 0.499 | 0.000 | 0.496 | 0.495 | 0.509 | 0.487 | 0.513 | 0.481 |
| C_lrhalf | 0.494 | 0.501 | 0.502 | 0.504 | 0.510 | 0.487 | 0.506 | 0.501 | 0.000 | 0.510 | 0.506 | 0.487 | 0.502 | 0.504 |
| C_onlynoise_seed | 0.508 | 0.501 | 0.510 | 0.492 | 0.485 | 0.504 | 0.505 | 0.495 | 0.510 | 0.000 | 0.061 | 0.494 | 0.489 | 0.042 |
| C_sameinit_diffdata | 0.513 | 0.494 | 0.499 | 0.515 | 0.474 | 0.500 | 0.499 | 0.497 | 0.503 | 0.059 | 0.000 | 0.498 | 0.486 | 0.067 |
| C_steps1500 | 0.491 | 0.489 | 0.501 | 0.498 | 0.494 | 0.491 | 0.498 | 0.484 | 0.488 | 0.497 | 0.506 | 0.000 | 0.505 | 0.496 |
| C_width2 | 0.491 | 0.491 | 0.498 | 0.499 | 0.500 | 0.496 | 0.518 | 0.512 | 0.502 | 0.503 | 0.499 | 0.511 | 0.000 | 0.500 |
| orig | 0.494 | 0.495 | 0.505 | 0.491 | 0.477 | 0.498 | 0.496 | 0.488 | 0.503 | 0.043 | 0.067 | 0.493 | 0.491 | 0.000 |

Mean diagonal BER 0.0000; mean off-diagonal BER 0.4832 (digital).

## Cost

```
{
 "enc_params": 1078337,
 "dec_params": 331200,
 "enc_bytes": 4317495,
 "dec_bytes": 1330367,
 "payload_bits": 32,
 "info_bits": 24,
 "bits_per_pixel": 0.0078125,
 "decode_ms_per_code_cpu1thread": 0.2632486820220947,
 "encode_ms_per_code_cpu1thread": 0.4242229461669922,
 "train_seconds_orig": 565.8980984687805
}
```