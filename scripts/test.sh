#!/bin/sh

SIZE=$1
SAVE_LOC=$2
DATA_LOC=$3
EVAL_BATCH$4
RUN_CKPT=$5

python train.py \
-train 0 \
-train_ckpt $DATA_LOC \
-save_location $SAVE_LOC \
-temp_size $SIZE \
-eval_batch_size $EVAL_BATCH \
-saved_ckpt $RUN_CKPT

