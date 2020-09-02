#!/usr/bin/env bash


# args: checkpoint_folder step

TASK=$1
PREFIX=$2
DIR=$3
CKP=$4
LR=$5
SEED=$6

BETAS="(0.9,0.999)"
CLIP=1.0
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
N_EPOCH=10


ARCH=roberta_large

SENT_PER_GPU=32
MAX_TOKENS=4400
UPDATE_FREQ=1

# valid 2 times per epoch
VALID_FREQ=2
ROOT=ft_log_0805

BERT_MODEL_PATH=$PREFIX/$DIR/$CKP

if [ ! -e $BERT_MODEL_PATH ]; then
    echo "Checkpoint doesn't exist"
    exit 0
fi

GLUE_DIR=glue-0508
DATA_DIR=$PREFIX/$GLUE_DIR/$TASK-bin-32768
OPTION=""

METRIC=accuracy
N_CLASSES=2

if [ "$TASK" = "MNLI" ]
then
N_CLASSES=3
OPTION="--valid-subset valid,valid1"
N_EPOCH=5
SENT_PER_GPU=16
MAX_TOKENS=2200
UPDATE_FREQ=2
EPOCH_ITER=12374
fi

if [ "$TASK" = "QNLI" ]
then
N_EPOCH=5
SENT_PER_GPU=16
MAX_TOKENS=2200
UPDATE_FREQ=2
EPOCH_ITER=3305
fi

if [ "$TASK" = "QQP" ]
then
N_EPOCH=5
SENT_PER_GPU=16
MAX_TOKENS=2200
UPDATE_FREQ=2
EPOCH_ITER=11382
fi

if ["$TASK" = "SST-2"]
then
EPOCH_ITER=2105
fi

if ["$TASK" = "MRPC"]
then
EPOCH_ITER=115
fi

if [ "$TASK" = "RTE" ]
then
SENT_PER_GPU=16
MAX_TOKENS=2200
UPDATE_FREQ=2
EPOCH_ITER=96
fi

if [ "$TASK" = "CoLA" ]
then
METRIC=mcc
EPOCH_ITER=268
WARMUP_RATIO=0.1
fi

if [ "$TASK" = "STS-B" ]
then
METRIC=pearson_spearman
N_CLASSES=1
OPTION="--regression-target"
EPOCH_ITER=180
WARMUP_RATIO=0.1
fi

TOTAL_STEPS=$((EPOCH_ITER*N_EPOCH))
WARMUP_STEPS=$((TOTAL_STEPS*WARMUP_RATIO))
echo $DATA_DIR
echo $TOTAL_STEPS
echo $WARMUP_STEPS

OUTPUT_PATH=$PREFIX/$ROOT/${DIR}/${CKP}/${TASK}/$LR-$SEED
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
if [ -e $OUTPUT_PATH/train_log.txt ]; then
    if grep -q 'done training' $OUTPUT_PATH/train_log.txt && grep -q 'loaded checkpoint' $OUTPUT_PATH/train_log.txt; then
        echo "Training log existed"
        exit 0
    fi
fi

python train.py $DATA_DIR --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --restore-file $BERT_MODEL_PATH \
    --max-positions 512 \
    --max-sentences $SENT_PER_GPU \
    --max-tokens $MAX_TOKENS \
    --update-freq $UPDATE_FREQ \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch $ARCH \
    --criterion sentence_prediction $OPTION \
    --num-classes $N_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "$BETAS" --adam-eps 1e-06 \
    --clip-norm $CLIP \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_STEPS --warmup-updates $WARMUP_STEPS  \
    --max-epoch $N_EPOCH --seed $SEED --save-dir $OUTPUT_PATH --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints --no-save \
    --find-unused-parameters --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir . \
    --best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric | tee $OUTPUT_PATH/train_log.txt 

