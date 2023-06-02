#!/bin/bash

DATASET=uspto_50k
SAVENAME=Model0
VOCAB=uspto_50k
SEED=17
K=4
KERNEL=spd
AVGNAME=AVG_MAIN


python average.py \
    --dataset_name="$DATASET" \
    --save_name="$SAVENAME" \
    --vocab_name="$VOCAB" \
    --seed="$SEED" \
    --K="$K" \
    --kernel="$KERNEL" \
    --ckpt_path="" \
    --average_list="" \
    --average_name="$AVGNAME"