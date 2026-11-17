#!/bin/bash
cd "$(dirname "$0")"
{ for i in 0 1 2 3 4; do echo "--name H$i --kind hidden --seed $i"; done
  for i in 0 1; do echo "--name K0_$i --kind key0 --seed $i"; echo "--name K1_$i --kind key1 --seed $i"; done
} | xargs -P 4 -I{} sh -c 'python train_h.py {} >> logs/train.log 2>&1'
