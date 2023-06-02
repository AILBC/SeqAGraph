#!/bin/bash

DATASET=uspto_full
SAVENAME=Model0
VOCAB=uspto_full
SEED=17
K=4
KERNEL=spd
BSZ=64
TOKENLIM=15000
D_MODEL=256
D_FILTER=2048
ENC_LAYER=6
DEC_LAYER=8
DROP=0.1
TASK=dualtask
ETASK=retrosynthesis
EVAL_START=500000
EVAL_STEP=2000
T=1.0
BEAM_GROUP=1
K_FILTER=0
P_FILTER=0.0
OPTIM=AdamW
LRSD=cosine
FFN=vanilla
NORM=rmsnorm
AUG_N=2

python train.py \
    --dataset_name="$DATASET" \
    --save_name="$SAVENAME" \
    --vocab_name="$VOCAB" \
    --seed="$SEED" \
    --K="$K" \
    --kernel="$KERNEL" \
    --batch_size="$BSZ" \
    --token_limit="$TOKENLIM" \
    --d_model="$D_MODEL" \
    --d_ff="$D_FILTER" \
    --enc_layer="$ENC_LAYER" \
    --dec_layer="$DEC_LAYER" \
    --dropout="$DROP" \
    --mode="train" \
    --task="$TASK" \
    --eval_task="$ETASK" \
    --epochs=1000 \
    --steps=600000 \
    --accum_count=2 \
    --lr_factor=1.0 \
    --max_lr=3e-4 \
    --min_lr=1e-6 \
    --warmup=10000 \
    --end_step=600000 \
    --gamma=2.0 \
    --train_eval \
    --eval_start="$EVAL_START" \
    --eval_step="$EVAL_STEP" \
    --split_data_len=10000 \
    --T="$T" \
    --beam_group="$BEAM_GROUP" \
    --top_k="$K_FILTER" \
    --top_p="$P_FILTER" \
    --optimizer="$OPTIM" \
    --lrschedule="$LRSD" \
    --ffn_type="$FFN" \
    --norm_type="$NORM" \
    --augment_N="$AUG_N"
