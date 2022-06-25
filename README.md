# Explaining Natural Language Inference with Factual and Template Memory Networks

## 1. Setup enviroment 
```bash
pip install -r requirements.txt
```
## 2. Download data from Google drive
```bash
gdown https://drive.google.com/file/d/1YDQR9Ob-zRGl7OcOXTxhNPmJa5fYatdL/view?usp=sharing
```

## 3. Run experiment with experiment script
The script feature the inputs for running the experiment
```bash
bash ./scripts/experiment.sh $SIZE $SAVE_LOC $DATA_LOC $EPOCHS $CEPOCHS $BATCH_SIZE $LR $CLR
```
To reproduce our paper result, we recommend 
```bash
bash ./scripts/experiment.sh 300 $SAVE_LOC data 10 10 32 3e-4 3e-6
```
## 4. Run test with test script
The script provides feature the inputs for running the testing
```bash
bash ./scripts/test.sh $SIZE $SAVE_LOC $DATA_LOC $EVAL_BATCH $RUN_CKPT
```
Evaluate the best checkpoint by changing the $RUN_CKPT,
```bash
bash ./scripts/test.sh 300 $SAVE_LOC data $EVAL_BATCH $RUN_CKPT
```
