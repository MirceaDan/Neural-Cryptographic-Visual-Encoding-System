#!/bin/bash
# same grid as NCVES attack: N in 25,100,500; 3 attacker seeds; target = H0 (HiDDeN-style) and K1_0/K0_0 (keyed) at N=100
cd "$(dirname "$0")"
{ for t in H0; do for n in 25 100 500; do for s in 0 1 2; do echo "--target $t --n $n --seed $s"; done; done; done
  for t in K0_0 K1_0; do for s in 0 1 2; do echo "--target $t --n 100 --seed $s"; done; done
} | xargs -P 3 -I{} sh -c 'python attack_h.py {} >> logs/attacks.log 2>&1'
