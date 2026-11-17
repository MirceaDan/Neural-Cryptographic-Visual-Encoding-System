#!/bin/bash
# Trains original + independent retrains (B) + variation conditions (C). 4 parallel jobs.
cd "$(dirname "$0")"
{
echo "--name orig --init_seed 0 --data_seed 0 --noise_seed 0"
for i in 1 2 3 4 5; do echo "--name B_r$i --init_seed $i --data_seed $i --noise_seed $i"; done
echo "--name C_sameinit_diffdata --init_seed 0 --data_seed 101 --noise_seed 101"
echo "--name C_diffinit_samedata --init_seed 102 --data_seed 0 --noise_seed 0"
echo "--name C_onlynoise_seed --init_seed 0 --data_seed 0 --noise_seed 103"
echo "--name C_lr2x --init_seed 104 --data_seed 104 --noise_seed 104 --lr 2e-3"
echo "--name C_lrhalf --init_seed 105 --data_seed 105 --noise_seed 105 --lr 5e-4"
echo "--name C_batch64 --init_seed 106 --data_seed 106 --noise_seed 106 --batch 64"
echo "--name C_steps1500 --init_seed 107 --data_seed 107 --noise_seed 107 --steps 1500"
echo "--name C_width2 --init_seed 108 --data_seed 108 --noise_seed 108 --width 2"
} | xargs -P 4 -I{} sh -c 'python train_instance.py {} --threads 3 >> logs/train.log 2>&1'
