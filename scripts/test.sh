#!/bin/sh

SIZE=900
SAVE_LOC=together_add_may_26_$SIZE
python train.py \
-train 0 \
-train_ckpt new_emb \
-save_location $SAVE_LOC \
-temp_size $SIZE \
-eval_batch_size 256 \
-saved_ckpt 3857

