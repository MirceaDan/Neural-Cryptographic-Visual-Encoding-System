#!/bin/bash
# Labeled substitute-decoder attacks (3 attacker seeds per N); warm-start and arch-mismatch variants.
cd "$(dirname "$0")"
{
for n in 25 100 500 2500 20000; do for s in 0 1 2; do echo "--n $n --seed $s"; done; done
for n in 25 100 500; do echo "--n $n --seed 0 --init_from B_r1"; done
echo "--n 2500 --seed 0 --width 2"
} | xargs -P 3 -I{} sh -c 'python attack_train.py {} >> logs/attacks.log 2>&1'
