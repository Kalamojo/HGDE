#!/bin/bash

lrs=(0.0005 0.001)
l1_reg=(0.00005)
hidden_size=(512 1024 2048 4096 8192 )

n=1
mkdir -p hyper_results
for hs in "${hidden_size[@]}"; do
    for reg in "${l1_reg[@]}"; do
        for lr in "${lrs[@]}"; do
            python3 run_benchmarks.py --model contrastive --eval contrastive --lr "$lr" --epochs 10 --hidden_size "$hs" --dropout 0.0 --batch_size 16 --l1_reg "$reg" --save_file "experiments/hyper_results/run_hidden_size${n}.json" --rounds 20
            ((n++))
        done
    done
done