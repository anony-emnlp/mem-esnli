#!/bin/sh


SIZE=900
SAVE_LOC=after_self_attention_$SIZE


python train.py \
-train 1 \
-train_ckpt new_emb \
-save_location $SAVE_LOC \
-temp_size $SIZE \
-train_batch_size 32 \
-epoch 10

python train.py \
-train 1 \
-c_train 1 \
-saved_ckpt 10 \
-train_ckpt new_emb \
-save_location $SAVE_LOC \
-temp_size $SIZE \
-train_batch_size 32 \
-learning_rate 3e-6 \
-epoch 10



python train.py \
    -train 0 \
    -train_ckpt new_emb \
    -eval_batch_size 256 \
    -temp_size $SIZE \
    -save_location $SAVE_LOC \
    -saved_ckpt 20
