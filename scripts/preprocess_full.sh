#!/bin/bash

DATASET=uspto_full
SAVENAME=Model0
VOCAB=uspto_full
SEED=17
K=4
KERNEL=spd
FILE_SPLIT=10000
EVAL_SPLIT=10000
AUG_N=2


python preprocess.py \
    --dataset_name="$DATASET" \
    --save_name="$SAVENAME" \
    --vocab_name="$VOCAB" \
    --seed="$SEED" \
    --K="$K" \
    --kernel="$KERNEL" \
    --split_data_len="$EVAL_SPLIT" \
    --smi2token \
    --tokenize \
    --featurize \
    --file_split="$FILE_SPLIT" \
    --augment_N="$AUG_N" \
    --split_shuffle