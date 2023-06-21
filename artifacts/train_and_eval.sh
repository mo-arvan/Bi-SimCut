

#export MODEL_DIR=model-save-dir-2
#export SEED=1315 # MAKE SURE to update

#EXP=iwslt14_de_en_simcut_alpha3_p005

EXP_BI=$EXP/bi
EXP_UNI=$EXP/uni

CHECKPOINT_DIR=$EXP_BI/checkpoints
LOG_DIR=$EXP_BI/logs

mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

START_TIME=`date +%s`

CUDA_VISIBLE_DEVICES=0 fairseq-train $IWSLT_BI_DATA_BIN \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_simcut --alpha 3.0 --p 0.05 --label-smoothing 0.1 \
    --max-tokens 4096 --no-epoch-checkpoints --save-dir $CHECKPOINT_DIR \
    --seed $SEED \
    1>$LOG_DIR/log.out 2>$LOG_DIR/log.err

RESULT=$?

if [ $RESULT -ne 0 ]; then
    echo "Training failed."
    cat $LOG_DIR/log.err
    exit
fi

END_TIME=`date +%s`

CKPT=$CHECKPOINT_DIR/checkpoint_best.pt

CHECKPOINT_DIR=$EXP_UNI/checkpoints
LOG_DIR=$EXP_UNI/logs

mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 fairseq-train $IWSLT_UNI_DATA_BIN \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_simcut --alpha 3.0 --p 0.05 --label-smoothing 0.1 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --restore-file $CKPT \
    --max-tokens 4096 --no-epoch-checkpoints --save-dir $CHECKPOINT_DIR \
    --seed $SEED \
    1>$LOG_DIR/log.out 2>$LOG_DIR/log.err


mkdir -p $EXP/evaluation

CKPT=$CHECKPOINT_DIR/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=0 fairseq-generate $IWSLT_UNI_DATA_BIN --path $CKPT \
    --gen-subset test --beam 5 --lenpen 1 --max-tokens 8192 --remove-bpe \
    > $EXP/evaluation


echo "Total time elapsed: $((END_TIME-START_TIME)) seconds" > $EXP/evaluation/elapsed.log