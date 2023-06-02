#!/bin/bash

DATASET=uspto_50k
SAVENAME=Model0
VOCAB=uspto_50k
SEED=17
K=4
KERNEL=spd
BSZ=64
TOKENLIM=0
D_MODEL=256
D_FILTER=2048
ENC_LAYER=6
DEC_LAYER=8
DROP=0.0
TASK=dualtask
ETASK=retrosynthesis
EMODE=test
BEAMSIZE=20
T=1.5
BEAM_GROUP=1
K_FILTER=0
P_FILTER=0.0
SEARCH_STEP=300
FFN=vanilla
NORM=rmsnorm
AUG_N=1


python train.py \
    --dataset_name="$DATASET" \
    --save_name="$SAVENAME" \
    --vocab_name="$VOCAB" \
    --seed="$SEED" \
    --K="$K" \
    --kernel="$KERNEL" \
    --eval_batch_size="$BSZ" \
    --eval_token_limit="$TOKENLIM" \
    --d_model="$D_MODEL" \
    --d_ff="$D_FILTER" \
    --enc_layer="$ENC_LAYER" \
    --dec_layer="$DEC_LAYER" \
    --dropout="$DROP" \
    --mode="eval" \
    --task="$TASK" \
    --eval_task="$ETASK" \
    --eval_mode="$EMODE" \
    --split_data_len=0 \
    --T="$T" \
    --beam_group="$BEAM_GROUP" \
    --top_k="$K_FILTER" \
    --top_p="$P_FILTER" \
    --beam_size="$BEAMSIZE" \
    --search_step="$SEARCH_STEP" \
    --ckpt_path="50k_class" \
    --ckpt_name="AVG_MAIN" \
    --ffn_type="$FFN" \
    --norm_type="$NORM" \
    --augment_N="$AUG_N" \
    --reaction_class