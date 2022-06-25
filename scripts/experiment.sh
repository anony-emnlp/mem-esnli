#!/bin/sh


SIZE=$1
SAVE_LOC=$2
DATA_LOC=$3
EPOCHS=$4
CEPOCHS=$5
BATCH_SIZE=$6
LR=$7
CLR=$8

python train.py \
-train 1 \
-train_ckpt $DATA_LOC \
-save_location $SAVE_LOC \
-temp_size $SIZE \
-learning_rate $LR \
-train_batch_size $BATCH_SIZE \
-epoch $EPOCHS

python train.py \
-train 1 \
-c_train 1 \
-saved_ckpt $EPOCHS \
-train_ckpt $DATA_LOC \
-save_location $SAVE_LOC \
-temp_size $SIZE \
-train_batch_size $BATCH_SIZE \
-learning_rate $CLR \
-epoch $CEPOCHS

