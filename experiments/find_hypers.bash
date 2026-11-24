#!/bin/bash

lrs=(0.00001 0.00005 0.0001 0.0005 0.001)
l1_reg=(0.00001 0.00005 0.0001 0.0005 0.001)

n=1
mkdir -p hyper_results

for reg in "${l1_reg[@]}"; do
    for lr in "${lrs[@]}"; do
        python3 run_benchmarks.py --model contrastive --eval contrastive --lr "$lr" --epochs 10 --hidden_size 512 --dropout 0.0 --batch_size 16 --l1_reg "$reg" --save_file "experiments/hyper_results/run${n}.json" --rounds 20
        ((n++))
    done
done