#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=16
DATA_PATH=/home/slu/Projects/metaGenerator/preprocess/metareview/metareview-bin
BART_PATH=/home/slu/Projects/metaGenerator/bart.large/model.pt
MODEL_PATH=/home/slu/Projects/metaGenerator/checkpoints
mkdir -p $MODEL_PATH
nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3 

fairseq-train $DATA_PATH \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --skip-invalid-size-inputs-valid-test \
    --batch-size 2 \
    --batch-size-valid 2\
    --required-batch-size-multiple 1 \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --save-dir $MODEL_PATH | tee -a $MODEL_PATH/train.log \